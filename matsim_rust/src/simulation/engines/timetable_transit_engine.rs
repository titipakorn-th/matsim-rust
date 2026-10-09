//! Runs configured transit services against their scheduled stop times.

use crate::simulation::Identifiable;
use crate::simulation::agents::agent::SimulationAgent;
use crate::simulation::agents::{
    AgentEvent, EndTime, EnvironmentalEventObserver, SimulationAgentLogic,
};
use crate::simulation::controller::ThreadLocalComputationalEnvironment;
use crate::simulation::events::{
    PersonArrivalEventBuilder, PersonDepartureEventBuilder, PersonEntersVehicleEventBuilder,
    PersonLeavesVehicleEventBuilder,
};
use crate::simulation::id::Id;
use crate::simulation::pt::driver::{StopOutcome, TransitDriver, serve_stop};
use crate::simulation::pt::runs::RunLeg;
use crate::simulation::pt::stops::TransitStops;
use crate::simulation::scenario::ScenarioCore;
use crate::simulation::scenario::network::Link;
use crate::simulation::scenario::network::Network;
use crate::simulation::scenario::population::InternalPerson;
use crate::simulation::scenario::transit::{TransitLine, TransitRoute, TransitSchedule};
use crate::simulation::scenario::vehicles::Garage;
use crate::simulation::time::{SimClock, SimTime, Tick};
use crate::simulation::time_queue::TimeQueue;
use crate::simulation::vehicles::SimulationVehicle;
use nohash_hasher::IntMap;
use nohash_hasher::IntSet;
use std::sync::Arc;
use std::time::Duration;

struct StopEvent {
    driver: Id<InternalPerson>,
    time: SimTime,
}

impl EndTime for StopEvent {
    fn end_time(&self, _now: SimTime) -> SimTime {
        self.time
    }
}

pub(crate) struct TimetableTransitEngine {
    waiting_drivers: TimeQueue<SimulationAgent, InternalPerson>,
    stop_events: TimeQueue<StopEvent, InternalPerson>,
    active: IntMap<Id<InternalPerson>, SimulationVehicle>,
    /// Vehicles whose next stop is served by another partition. The caller hands them over
    /// through the network message broker, exactly as a vehicle leaving a link does.
    migrating: Vec<SimulationVehicle>,
    schedule: Arc<TransitSchedule>,
    deterministic_modes: IntSet<Id<String>>,
    garage: Arc<Garage>,
    network: Arc<Network>,
    rank: u32,
    comp_env: ThreadLocalComputationalEnvironment,
    clock: SimClock,
}

impl TimetableTransitEngine {
    pub(crate) fn new(
        scenario: &ScenarioCore,
        rank: u32,
        comp_env: ThreadLocalComputationalEnvironment,
        clock: SimClock,
        start: SimTime,
    ) -> Self {
        let mut engine = Self {
            waiting_drivers: TimeQueue::new(),
            stop_events: TimeQueue::new(),
            active: IntMap::default(),
            migrating: Vec::new(),
            schedule: scenario.transit_schedule.clone(),
            deterministic_modes: scenario
                .config
                .transit()
                .deterministic_service_modes
                .iter()
                .map(|mode| Id::get_from_ext(mode))
                .collect(),
            garage: scenario.garage.clone(),
            network: scenario.network.clone(),
            rank,
            comp_env,
            clock,
        };
        // A timetable vehicle serves its first stop from the moment it departs, so it belongs to
        // the partition owning that stop's link rather than to the one owning the route's first
        // link, which may be further upstream and on another partition.
        for run in scenario
            .transit_runs
            .runs()
            .iter()
            .filter(|run| run.is_deterministic())
        {
            let RunLeg::Service {
                route,
                departure_time,
                ..
            } = &run.legs[0]
            else {
                unreachable!("timetable services have no deadheads")
            };
            let Some(first_stop) = route.stops.first() else {
                continue;
            };
            if scenario.network.get_link(&first_stop.link).partition != rank {
                continue;
            }
            let driver =
                SimulationAgent::new(Box::new(TransitDriver::new(run.clone(), &engine.garage)));
            engine.waiting_drivers.add_with_order(
                driver,
                start.max(*departure_time),
                run.driver.internal(),
            );
        }
        engine
    }

    pub(crate) fn serves(&self, agent: &SimulationAgent) -> bool {
        let Some(pt_route) = agent
            .curr_leg()
            .route
            .as_ref()
            .and_then(|route| route.as_pt())
        else {
            return false;
        };
        let Some(line_id) =
            Id::<TransitLine>::try_get_from_ext(&pt_route.description.transit_line_id)
        else {
            return false;
        };
        let Some(route_id) =
            Id::<TransitRoute>::try_get_from_ext(&pt_route.description.transit_route_id)
        else {
            return false;
        };
        self.schedule
            .lines()
            .get(&line_id)
            .and_then(|line| line.routes.get(&route_id))
            .is_some_and(|route| self.deterministic_modes.contains(&route.transport_mode))
    }

    pub(crate) fn drain(&mut self) -> Vec<SimulationAgent> {
        self.waiting_drivers
            .drain()
            .into_iter()
            .chain(
                self.active
                    .drain()
                    .flat_map(|(_, vehicle)| vehicle.into_agents()),
            )
            .collect()
    }

    /// Vehicles whose next stop is served by another partition, drained by the caller.
    pub(crate) fn take_migrating(&mut self) -> Vec<SimulationVehicle> {
        std::mem::take(&mut self.migrating)
    }

    /// Whether this engine drives the vehicle. A timetable vehicle never joins the queue network
    /// engine, so a vehicle arriving from another partition comes back here.
    pub(crate) fn drives(&self, vehicle: &SimulationVehicle) -> bool {
        vehicle
            .driver()
            .transit_driver()
            .is_some_and(|driver| driver.run().is_deterministic())
    }

    /// Takes over a vehicle another partition handed over and serves the stop it came for.
    pub(crate) fn receive_vehicle(
        &mut self,
        now: Tick,
        vehicle: SimulationVehicle,
        stops: &mut TransitStops,
    ) -> Vec<SimulationAgent> {
        let now = self.clock.tick_to_time(now);
        let key = vehicle.driver().id().clone();
        self.place(vehicle, key, now, false);
        // The vehicle arrived through this step's messages, after this engine processed its own
        // stop events, so a stop that is already due is served right away.
        let mut completed = Vec::new();
        self.serve_due_stops(now, stops, &mut completed);
        completed
    }

    pub(crate) fn do_step(&mut self, now: Tick, stops: &mut TransitStops) -> Vec<SimulationAgent> {
        let now = self.clock.tick_to_time(now);
        let mut completed = Vec::new();
        for mut driver in self.waiting_drivers.pop(now) {
            driver.advance_plan(now);
            let vehicle_id = driver.transit_driver().unwrap().run().vehicle.clone();
            {
                let mut events = self.comp_env.events_manager_borrow_mut();
                driver
                    .transit_driver()
                    .unwrap()
                    .emit_starts(now, &mut events);
                let leg = driver.curr_leg();
                events.process_event(
                    &PersonDepartureEventBuilder::default()
                        .time(now)
                        .person(driver.id().clone())
                        .link(driver.curr_link_id().unwrap().clone())
                        .leg_mode(leg.mode.clone())
                        .routing_mode(leg.routing_mode.clone().unwrap())
                        .build()
                        .unwrap(),
                );
                events.process_event(
                    &PersonEntersVehicleEventBuilder::default()
                        .time(now)
                        .person(driver.id().clone())
                        .vehicle(vehicle_id.clone())
                        .build()
                        .unwrap(),
                );
            }
            let key = driver.id().clone();
            let vehicle = self.garage.unpark_veh(driver, vehicle_id);
            self.place(vehicle, key, now, true);
        }
        self.serve_due_stops(now, stops, &mut completed);
        completed
    }

    fn serve_due_stops(
        &mut self,
        now: SimTime,
        stops: &mut TransitStops,
        completed: &mut Vec<SimulationAgent>,
    ) {
        for event in self.stop_events.pop(now) {
            let event_time = event.time;
            let Some(mut vehicle) = self.active.remove(&event.driver) else {
                continue;
            };
            let link = vehicle.curr_link_id().unwrap().clone();
            let outcome = {
                let mut events = self.comp_env.events_manager_borrow_mut();
                serve_stop(&mut vehicle, &link, event_time, stops, &mut events, true)
            };
            match outcome {
                StopOutcome::Dwell { seconds, .. } => {
                    let next = event_time.saturating_add(Duration::from_secs_f64(seconds));
                    self.reschedule(event.driver.clone(), next);
                    self.active.insert(event.driver, vehicle);
                }
                StopOutcome::Departed => {
                    if vehicle.driver().transit_driver().unwrap().is_finished() {
                        self.finish_vehicle(event_time, vehicle, completed);
                    } else {
                        self.place(vehicle, event.driver, event_time, false);
                    }
                }
                StopOutcome::NoStop => panic!("A timetable event must always target a route stop."),
            }
        }
    }

    /// Moves the vehicle onto the link of its next stop and schedules that stop. The stop belongs to
    /// the partition owning its link, which is where the passengers waiting for it are, so the
    /// vehicle is kept back for migration when that is another partition.
    fn place(
        &mut self,
        mut vehicle: SimulationVehicle,
        key: Id<InternalPerson>,
        now: SimTime,
        first: bool,
    ) {
        let Some(stop_link) = vehicle
            .driver()
            .transit_driver()
            .unwrap()
            .next_stop_link()
            .cloned()
        else {
            // The run has no stop left. The vehicle stays here until the simulation ends, which is
            // where it ended before vehicles moved between partitions at all.
            self.active.insert(key, vehicle);
            return;
        };
        drive_to_link(&mut vehicle, &stop_link, now);
        if self.network.get_link(&stop_link).partition != self.rank {
            self.migrating.push(vehicle);
            return;
        }
        let time = self.stop_time(&vehicle, now, first);
        self.reschedule(key.clone(), time);
        self.active.insert(key, vehicle);
    }

    /// When the next stop is due. The first stop is served as the vehicle departs, a later one at
    /// its scheduled arrival, and never before the vehicle left the previous stop.
    fn stop_time(&self, vehicle: &SimulationVehicle, now: SimTime, first: bool) -> SimTime {
        if first {
            return now;
        }
        let transit_driver = vehicle.driver().transit_driver().unwrap();
        let RunLeg::Service {
            route,
            departure_time,
            ..
        } = transit_driver.run().legs.last().unwrap()
        else {
            unreachable!("timetable services have no deadheads")
        };
        let stop = &route.stops[transit_driver.next_stop_index()];
        let offset = stop
            .arrival_offset
            .or(stop.departure_offset)
            .unwrap_or_default();
        now.max(departure_time.saturating_add(offset))
    }

    fn reschedule(&mut self, driver: Id<InternalPerson>, time: SimTime) {
        self.stop_events.add_with_order(
            StopEvent {
                driver: driver.clone(),
                time,
            },
            time,
            driver.internal(),
        );
    }

    fn finish_vehicle(
        &mut self,
        now: SimTime,
        mut vehicle: SimulationVehicle,
        completed: &mut Vec<SimulationAgent>,
    ) {
        let mut events = self.comp_env.events_manager_borrow_mut();
        events.process_event(
            &PersonLeavesVehicleEventBuilder::default()
                .time(now)
                .person(vehicle.driver().id().clone())
                .vehicle(vehicle.id().clone())
                .build()
                .unwrap(),
        );
        events.process_event(
            &PersonArrivalEventBuilder::default()
                .time(now)
                .person(vehicle.driver().id().clone())
                .link(vehicle.curr_link_id().unwrap().clone())
                .leg_mode(vehicle.driver().curr_leg().mode.clone())
                .build()
                .unwrap(),
        );
        vehicle.driver_mut().advance_plan(now);
        completed.extend(vehicle.into_agents());
    }
}

/// Advances the vehicle's driver over the intermediate links to the stop it serves next. A
/// timetable vehicle does not occupy the network between stops, so this is route bookkeeping only.
fn drive_to_link(vehicle: &mut SimulationVehicle, link: &Id<Link>, now: SimTime) {
    while vehicle.curr_link_id() != Some(link) {
        assert!(
            vehicle.peek_next_route_element().is_some(),
            "Transit route ends before its next scheduled stop on link {link}."
        );
        vehicle.notify_event(&mut AgentEvent::LeftLink(), now);
    }
}
