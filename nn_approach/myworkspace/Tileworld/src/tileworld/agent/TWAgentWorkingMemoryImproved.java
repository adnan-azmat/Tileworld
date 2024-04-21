package tileworld.agent;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.Map;
import java.util.AbstractMap.SimpleEntry;
import java.util.Collections;
import java.util.Comparator;

import javax.swing.text.html.HTMLDocument;
import sim.engine.Schedule;
import sim.field.grid.ObjectGrid2D;
import sim.util.Bag;
import sim.util.Double2D;
import sim.util.Int2D;
import sim.util.IntBag;
import tileworld.environment.NeighbourSpiral;
import tileworld.Parameters;
import tileworld.environment.TWEntity;

import tileworld.environment.TWHole;
import tileworld.environment.TWObject;
import tileworld.environment.TWObstacle;
import tileworld.environment.TWTile;
import tileworld.environment.TWFuelStation;

public class TWAgentWorkingMemoryImproved extends TWAgentWorkingMemory {

	private Schedule schedule;
	private TWAgent me;
	private final static int MAX_TIME = 10;
	private final static float MEM_DECAY = 0.5f;
	private ObjectGrid2D memoryGrid;
	private TWAgentPercept[][] objects;
	private int memorySize; // TODO is this variable accessed or the superclass one?
	static private List<Int2D> spiral = new NeighbourSpiral(Parameters.defaultSensorRange * 4).spiral();
	static private List<Int2D> spiralMemory = new NeighbourSpiral(Parameters.defaultSensorRange * 9).spiral();

	private Set<TWAgentPercept> fuelStations;
	private Set<TWAgentPercept> agents;

	private Bag sensedObjects;
	private Bag sensedAgents;
	private double sensedTime;

	public TWAgentWorkingMemoryImproved(TWAgent moi, Schedule schedule, int x, int y) {
		super(moi, schedule, x, y);
		this.objects = new TWAgentPercept[x][y];
		this.schedule = schedule;
		this.me = moi;
		this.memoryGrid = new ObjectGrid2D(me.getEnvironment().getxDimension(), me.getEnvironment().getyDimension());

		this.sensedAgents = new Bag();
		this.sensedObjects = new Bag();
		this.sensedTime = 0.0;

		this.fuelStations = new HashSet<>();
		this.agents = new HashSet<>();
	}

	// Example of overriding a method to enhance or change its behavior
	@Override
	public void updateMemory(Bag sensedObjects, IntBag objectXCoords, IntBag objectYCoords, Bag sensedAgents,
			IntBag agentXCoords, IntBag agentYCoords) {
		decayMemory(); // removes old observations (by threshold)

		updateMemoryAt(me.getX(), me.getY(), sensedObjects, sensedAgents, this.getSimulationTime());
	}

	public void updateMemoryAt(Integer xAgent, Integer yAgent, Bag sensedObjects, Bag sensedAgents, Double simTime) {

		// save sensed things
		this.sensedAgents = sensedAgents;
		this.sensedObjects = sensedObjects;
		this.sensedTime = simTime;

		// Remove old objects
		for (Int2D offset : spiral) {
			int x = offset.x + xAgent;
			int y = offset.y + yAgent;
			if (me.getEnvironment().isInBounds(x, y)) {
				if (objects[x][y] != null)
					memorySize--;
				objects[x][y] = null;
				memoryGrid.set(x, y, null);
			}
		}

		// Write new TWObjects
		for (int i = 0; i < sensedObjects.size(); i++) {
			TWEntity o = (TWEntity) sensedObjects.get(i);
			if (o instanceof TWObject) {
				if (objects[o.getX()][o.getY()] == null)
					memorySize++;
				objects[o.getX()][o.getY()] = new TWAgentPercept(o, simTime);
				getMemoryGrid().set(o.getX(), o.getY(), o);
			} else if (o instanceof TWFuelStation) {
				// add to fuel station list
				System.out.println("Found Fuel Station Yaaaaay");
			}
		}

		// Update TW Agents set
		for (int i = 0; i < sensedObjects.size(); i++) {
			TWEntity a = (TWEntity) sensedObjects.get(i);
			if (a instanceof TWAgent) {
				TWAgent agent = (TWAgent) a;
				TWAgentPercept newPercept = new TWAgentPercept(agent, simTime);

				TWAgentPercept existingPercept = null;
				for (TWAgentPercept percept : agents) {
					if (percept.getO().equals(agent)) {
						existingPercept = percept;
						break;
					}
				}
				if (existingPercept != null) {
					agents.remove(existingPercept);
				}
				agents.add(newPercept);
			}
		}
	}

	public void decayMemory() {
		for (int x = 0; x < this.objects.length; x++) {
			for (int y = 0; y < this.objects[x].length; y++) {
				TWAgentPercept currentMemory = objects[x][y];
				if (currentMemory != null && currentMemory.getT() < schedule.getTime() - MAX_TIME) {
					memorySize--;
					objects[x][y] = null;
					memoryGrid.set(x, y, null);
				}
			}
		}
	}

	public Map<String, Map<Int2D, Double2D>> getNearbyPercepts() {
		int MAX_ENTRIES = 5;
		int xAgent = me.getX();
		int yAgent = me.getY();

		List<Map.Entry<Int2D, Double>> obstacles = new ArrayList<>();
		List<Map.Entry<Int2D, Double>> tiles = new ArrayList<>();
		List<Map.Entry<Int2D, Double>> holes = new ArrayList<>();

		// Remove old objects
		for (Int2D offset : spiralMemory) {
			int x = offset.x + xAgent;
			int y = offset.y + yAgent;
			if (me.getEnvironment().isInBounds(x, y)) {
				if (getMemoryGrid().get(x, y) != null) {

					TWEntity o = (TWEntity) getMemoryGrid().get(x, y);
					if (o instanceof TWObject) {
						if (objects[o.getX()][o.getY()] == null)
							memorySize++;

						double distance = me.getDistanceTo(x, y);

						if (o instanceof TWHole) {
							holes.add(new SimpleEntry<>(new Int2D(offset.x, offset.y), distance));
						}
						if (o instanceof TWObstacle) {
							obstacles.add(new SimpleEntry<>(new Int2D(offset.x, offset.y), distance));
						}
						if (o instanceof TWTile) {
							tiles.add(new SimpleEntry<>(new Int2D(offset.x, offset.y), distance));
						} 
					}
				}
			}
		}

		Map<String, Map<Int2D, Double2D>> topObstacles = processEntries(obstacles, MAX_ENTRIES);
		Map<String, Map<Int2D, Double2D>> topTiles = processEntries(tiles, MAX_ENTRIES);
		Map<String, Map<Int2D, Double2D>> topHoles = processEntries(holes, MAX_ENTRIES);

		Map<String, Map<Int2D, Double2D>> top5 = new HashMap<>();
		top5.put("obstacles", topObstacles.get("entries"));
		top5.put("tiles", topTiles.get("entries"));
		top5.put("holes", topHoles.get("entries"));

		return top5;
	}

	public Map<String, Map<Int2D, Double2D>> processEntries(List<Map.Entry<Int2D, Double>> list, int MAX_ENTRIES) {
		Collections.sort(list, Comparator.comparingDouble(Map.Entry::getValue));
		List<Map.Entry<Int2D, Double>> topEntries = list.subList(0, Math.min(list.size(), MAX_ENTRIES));

		Map<String, Map<Int2D, Double2D>> results = new HashMap<>();
		Map<Int2D, Double2D> magnitudeAngleMap = new HashMap<>();

		for (Map.Entry<Int2D, Double> entry : topEntries) {
			Int2D location = entry.getKey();
			double magnitude = calculateManhattanMagnitude(location.x, location.y);
			double angle = calculateAngle(location.x, location.y);

			magnitudeAngleMap.put(new Int2D(location.x, location.y), new Double2D(magnitude, angle));
		}

		results.put("entries", magnitudeAngleMap);
		return results;
	}

	private static double calculateManhattanMagnitude(double x, double y) {
		return Math.abs(x) + Math.abs(y);
	}

	private static double calculateAngle(double x, double y) {
		return Math.atan2(y, x);
	}

	public void setSensedObjects(Bag sensedObjects) {
		this.sensedObjects = sensedObjects;
	}

	public void setSensedTime(Double sensedTime) {
		this.sensedTime = sensedTime;
	}

	public Bag getSensedAgents() {
		return this.sensedAgents;
	}

	public Bag getSensedObjects() {
		return this.sensedObjects;
	}

	public Double getSensedTime() {
		return this.sensedTime;
	}

	private double getSimulationTime() {
		return schedule.getTime();
	}

	public void addFuelStation(TWFuelStation fuelStation) {
		this.fuelStations.add(new TWAgentPercept(fuelStation, this.getSimulationTime()));
	}

	public Set<TWAgentPercept> getFuelStations() {
		return new HashSet<>(this.fuelStations); // Return a copy to prevent external modification
	}

	public boolean hasFuelStations() {
		return !fuelStations.isEmpty();
	}

}