use crate::simulation::Identifiable;
use crate::simulation::events::{
    ActivityEndEvent, ActivityStartEvent, LinkEnterEvent, PersonArrivalEvent, PersonDepartureEvent,
    PersonEntersVehicleEvent, PersonLeavesVehicleEvent, PersonStuckEvent,
    PtTeleportationArrivalEvent, TeleportationArrivalEvent, VehicleEntersTrafficEvent,
    VehicleLeavesTrafficEvent,
};
use crate::simulation::id::Id;
use crate::simulation::messaging::partition_change::PartitionChangeEntity;
use crate::simulation::pt::runs::TransitVehicleRuns;
use crate::simulation::scenario::population::InternalPerson;
use crate::simulation::scenario::vehicles::InternalVehicle;
use crate::simulation::scoring::backpacking::backpack::{Backpack, PersonExperience};
use crate::simulation::scoring::partial_plans::PartialPlan;
use nohash_hasher::{IntMap, IntSet};
use std::sync::Arc;

pub(crate) struct BackpackingAttachment {
    backpacks: Vec<Backpack>,
}

pub struct BackpackingDataCollector {
    home_person_ids: Vec<Id<InternalPerson>>,
    transit_runs: Arc<TransitVehicleRuns>,
    person_id2backpack: IntMap<Id<InternalPerson>, Backpack>,
    vehicle_id2person_ids: IntMap<Id<InternalVehicle>, IntSet<Id<InternalPerson>>>,
    person_id2vehicle_id: IntMap<Id<InternalPerson>, Id<InternalVehicle>>,
}

impl BackpackingDataCollector {
    pub fn new(
        home_person_ids: Vec<Id<InternalPerson>>,
        transit_runs: Arc<TransitVehicleRuns>,
    ) -> Self {
        Self {
            home_person_ids,
            transit_runs,
            person_id2backpack: Default::default(),
            vehicle_id2person_ids: Default::default(),
            person_id2vehicle_id: Default::default(),
        }
    }

    #[hotpath::measure]
    pub(crate) fn reset_iteration(&mut self) {
        self.person_id2backpack.clear();
        for person in &self.home_person_ids {
            self.person_id2backpack
                .insert(person.clone(), Backpack::new(person.clone()));
        }
        self.vehicle_id2person_ids.clear();
        self.person_id2vehicle_id.clear();
    }

    /// Records the person as an occupant of the vehicle and forwards the event to its backpack.
    #[hotpath::measure]
    pub(crate) fn person_enters_vehicle(&mut self, event: &PersonEntersVehicleEvent) {
        if self.transit_runs.is_driver(&event.person) {
            return;
        }
        let previous = self
            .person_id2vehicle_id
            .insert(event.person.clone(), event.vehicle.clone());
        assert!(previous.is_none() || previous.as_ref() == Some(&event.vehicle));
        self.vehicle_id2person_ids
            .entry(event.vehicle.clone())
            .or_default()
            .insert(event.person.clone());
        self.partial_plan(&event.person)
            .person_enters_vehicle(event);
    }

    /// Removes the person from the occupants of the vehicle and forwards the event to its backpack.
    #[hotpath::measure]
    pub(crate) fn person_leaves_vehicle(&mut self, event: &PersonLeavesVehicleEvent) {
        if self.transit_runs.is_driver(&event.person) {
            return;
        }
        self.remove_vehicle_membership(&event.person);
        if let Some(plan) = self.partial_plan_for_person(&event.person) {
            plan.person_leaves_vehicle(event);
        }
    }

    pub(crate) fn person_departure(&mut self, event: &PersonDepartureEvent) {
        if let Some(plan) = self.partial_plan_for_person(&event.person) {
            plan.person_departure(event);
        }
    }

    pub(crate) fn person_arrival(&mut self, event: &PersonArrivalEvent) {
        if let Some(plan) = self.partial_plan_for_person(&event.person) {
            plan.person_arrival(event);
        }
    }

    pub(crate) fn activity_start(&mut self, event: &ActivityStartEvent) {
        if let Some(plan) = self.partial_plan_for_person(&event.person) {
            plan.activity_start(event);
        }
    }

    pub(crate) fn activity_end(&mut self, event: &ActivityEndEvent) {
        if let Some(plan) = self.partial_plan_for_person(&event.person) {
            plan.activity_end(event);
        }
    }

    pub(crate) fn teleportation_arrival(&mut self, event: &TeleportationArrivalEvent) {
        if let Some(plan) = self.partial_plan_for_person(&event.person) {
            plan.teleportation_arrival(event);
        }
    }

    pub(crate) fn pt_teleportation_arrival(&mut self, event: &PtTeleportationArrivalEvent) {
        if let Some(plan) = self.partial_plan_for_person(&event.person) {
            plan.pt_teleportation_arrival(event);
        }
    }

    pub(crate) fn person_stuck(&mut self, event: &PersonStuckEvent) {
        if let Some(plan) = self.partial_plan_for_person(&event.person) {
            plan.person_stuck(event);
        }
        self.remove_vehicle_membership(&event.person);
    }

    pub(crate) fn link_enter(&mut self, event: &LinkEnterEvent) {
        self.for_each_occupant(&event.vehicle, |plan| plan.link_enter(event));
    }

    pub(crate) fn vehicle_enters_traffic(&mut self, event: &VehicleEntersTrafficEvent) {
        self.for_each_occupant(&event.vehicle, |plan| plan.vehicle_enters_traffic(event));
    }

    pub(crate) fn vehicle_leaves_traffic(&mut self, event: &VehicleLeavesTrafficEvent) {
        self.for_each_occupant(&event.vehicle, |plan| plan.vehicle_leaves_traffic(event));
    }

    /// Passes an event of a vehicle to the backpacks of all its occupants. Events of vehicles
    /// without occupants on this partition are ignored.
    fn for_each_occupant(
        &mut self,
        vehicle: &Id<InternalVehicle>,
        mut handle: impl FnMut(&mut PartialPlan),
    ) {
        let Some(persons) = self.vehicle_id2person_ids.get(vehicle) else {
            return;
        };
        for person in persons {
            if self.transit_runs.is_driver(person) {
                continue;
            }
            handle(Self::backpack(&mut self.person_id2backpack, person).partial_plan_mut());
        }
    }

    fn partial_plan_for_person(&mut self, person: &Id<InternalPerson>) -> Option<&mut PartialPlan> {
        if self.transit_runs.is_driver(person) {
            return None;
        }
        Some(self.partial_plan(person))
    }

    fn partial_plan(&mut self, person: &Id<InternalPerson>) -> &mut PartialPlan {
        Self::backpack(&mut self.person_id2backpack, person).partial_plan_mut()
    }

    fn remove_vehicle_membership(&mut self, person: &Id<InternalPerson>) {
        let Some(vehicle) = self.person_id2vehicle_id.remove(person) else {
            return;
        };
        let remove_vehicle = self
            .vehicle_id2person_ids
            .get_mut(&vehicle)
            .map(|persons| {
                persons.remove(person);
                persons.is_empty()
            })
            .unwrap_or(false);
        if remove_vehicle {
            self.vehicle_id2person_ids.remove(&vehicle);
        }
    }

    fn backpack<'a>(
        backpacks: &'a mut IntMap<Id<InternalPerson>, Backpack>,
        person: &Id<InternalPerson>,
    ) -> &'a mut Backpack {
        backpacks.get_mut(person).unwrap_or_else(|| {
            panic!(
                "No backpack is available for person {} while handling an event.",
                person.external()
            )
        })
    }

    #[hotpath::measure]
    pub(crate) fn send(&mut self, entity: PartitionChangeEntity<'_>) -> BackpackingAttachment {
        let person_ids = self.person_ids(entity);
        if let PartitionChangeEntity::Vehicle(vehicle) = entity {
            self.vehicle_id2person_ids.remove(vehicle.id());
            for person in &person_ids {
                self.person_id2vehicle_id.remove(person);
            }
        }

        let backpacks = person_ids
            .into_iter()
            .map(|person_id| {
                self.person_id2backpack
                    .remove(&person_id)
                    .unwrap_or_else(|| {
                        panic!(
                            "No backpack is available for departing person {}.",
                            person_id.external()
                        )
                    })
            })
            .collect();
        BackpackingAttachment { backpacks }
    }

    #[hotpath::measure]
    pub(crate) fn receive(
        &mut self,
        entity: PartitionChangeEntity<'_>,
        attachment: BackpackingAttachment,
    ) {
        let expected_person_ids = self.person_ids(entity);
        assert_eq!(
            attachment.backpacks.len(),
            expected_person_ids.len(),
            "Backpacking attachment contains the wrong number of backpacks."
        );

        for (expected_id, backpack) in expected_person_ids.iter().zip(attachment.backpacks) {
            assert_eq!(
                backpack.person_id(),
                expected_id,
                "Backpacking attachment contains a backpack for the wrong person."
            );
            let previous = self
                .person_id2backpack
                .insert(expected_id.clone(), backpack);
            assert!(
                previous.is_none(),
                "A backpack for arriving person {} is already present.",
                expected_id.external()
            );
            if let PartitionChangeEntity::Vehicle(vehicle) = entity {
                let previous = self
                    .person_id2vehicle_id
                    .insert(expected_id.clone(), vehicle.id().clone());
                assert!(
                    previous.is_none(),
                    "Arriving person already occupies a vehicle."
                );
            }
        }

        if let PartitionChangeEntity::Vehicle(vehicle) = entity {
            let persons = expected_person_ids.into_iter().collect();
            let previous = self
                .vehicle_id2person_ids
                .insert(vehicle.id().clone(), persons);
            assert!(
                previous.is_none(),
                "A vehicle mapping for arriving vehicle {} is already present.",
                vehicle.id().external()
            );
        }
    }

    fn person_ids(&self, entity: PartitionChangeEntity<'_>) -> Vec<Id<InternalPerson>> {
        match entity {
            PartitionChangeEntity::Vehicle(vehicle) => std::iter::once(vehicle.driver().id())
                .chain(vehicle.passengers().iter().map(Identifiable::id))
                .filter(|person| !self.transit_runs.is_driver(person))
                .cloned()
                .collect(),
            PartitionChangeEntity::TeleportationAgent(agent) => {
                if self.transit_runs.is_driver(agent.id()) {
                    Vec::new()
                } else {
                    vec![agent.id().clone()]
                }
            }
        }
    }

    #[hotpath::measure]
    pub(crate) fn finish(&mut self) -> IntMap<Id<InternalPerson>, PersonExperience> {
        self.person_id2backpack
            .drain()
            .map(|(person_id, backpack)| (person_id, backpack.finish()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::scenario::vehicles::InternalVehicle;
    use crate::simulation::vehicles::SimulationVehicle;
    use crate::test_utils::create_agent;
    use macros::deterministic_id_test;

    #[deterministic_id_test]
    fn migrates_driver_and_passenger_and_rebuilds_vehicle_mapping() {
        let driver = create_agent(1, vec!["destination"]);
        let passenger = create_agent(2, vec!["destination"]);
        let driver_id = driver.id().clone();
        let passenger_id = passenger.id().clone();
        let vehicle = SimulationVehicle::new(
            InternalVehicle::new(10, 0, 1.0, 1.0),
            Some(driver),
            vec![passenger],
        );
        let transit_runs = Arc::new(TransitVehicleRuns::default());
        let mut departing = BackpackingDataCollector::new(
            vec![driver_id.clone(), passenger_id.clone()],
            transit_runs.clone(),
        );
        departing.reset_iteration();
        departing.vehicle_id2person_ids.insert(
            vehicle.id().clone(),
            [driver_id.clone(), passenger_id.clone()]
                .into_iter()
                .collect(),
        );

        let attachment = departing.send(PartitionChangeEntity::Vehicle(&vehicle));
        assert!(departing.person_id2backpack.is_empty());
        assert!(departing.vehicle_id2person_ids.is_empty());

        let mut arriving = BackpackingDataCollector::new(Vec::new(), transit_runs);
        arriving.receive(PartitionChangeEntity::Vehicle(&vehicle), attachment);
        assert_eq!(arriving.person_id2backpack.len(), 2);
        let expected: IntSet<_> = [driver_id, passenger_id].into_iter().collect();
        assert_eq!(
            arriving.vehicle_id2person_ids.get(vehicle.id()).unwrap(),
            &expected
        );
    }
}
