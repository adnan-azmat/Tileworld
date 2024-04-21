package tileworld.agent;

import java.awt.Color;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.PriorityQueue;

import sim.display.GUIState;
import sim.portrayal.Inspector;
import sim.portrayal.LocationWrapper;
import sim.portrayal.Portrayal;
import sim.util.Bag;
import sim.util.Int2D;
import sim.util.IntBag;
import tileworld.Parameters;
import tileworld.agent.MyAgent.Mode;
import tileworld.environment.TWDirection;
import tileworld.environment.TWEntity;
import tileworld.environment.TWEnvironment;
import tileworld.environment.TWFuelStation;
import tileworld.environment.TWHole;
import tileworld.environment.TWTile;
import tileworld.exceptions.CellBlockedException;
import tileworld.planners.DefaultTWPlanner;

/*
 * use createAgent(new MyInspectorAgent(zoneIndentity, "agentInspector 0", pos.getX(), pos.getY(), this, Parameters.defaultFuelLevel));
 * 	zoneIndentity range: [0,1,2,3]
 * */

public class MyInspectorAgent extends TWAgent {
	
	/* 
	 * EXPLORE: random walking, when no Tiles observed in the agents memory
	 * COLLECT: fuel is adequate, tiles < 3, have position 
	 * FILL: tiles = 3, have 
	 * REFUEL: fuel below a threshold, already know the fuel station
	 * WAIT: fuel below a threshold, wait other agent found fuel station 
	 * */
	enum Mode {
		EXPLORE, COLLECT, FILL, REFUEL, WAIT, FIND_FUEL_STATION, INSPECT
	}
	
	private int index;
	private String name;
	private MyMemory memory;
	private DefaultTWPlanner planner;
	private Mode mode;
	private double fuelThreshold;
	
	TWEntity neighborTile;
	TWEntity neighborHole;
	
	PriorityQueue<TWEntity> neighborTiles;
	PriorityQueue<TWEntity> neighborHoles;
	
	
	public MyInspectorAgent(int index, String name, int xpos, int ypos, TWEnvironment env, double fuelLevel) {
        super(xpos,ypos,env,fuelLevel);
        this.index = index;
        this.name = name;
        this.fuelThreshold = this.getFuelLevel() / 10 ;
        this.planner = new DefaultTWPlanner(this);
		this.sensor = new TWAgentSensor(this, Parameters.defaultSensorRange);
        this.memory = new MyMemory(this, env.schedule, env.getxDimension(), env.getyDimension());
//		this.memory = (MyMemory) this.memory;
	}
	
	
	public MyInspectorAgent(int xpos, int ypos, TWEnvironment env, double fuelLevel) {
		super(xpos, ypos, env, fuelLevel);
		this.planner = new DefaultTWPlanner(this);
	}
	
	
	// get position FuelStation, and spread its if this agent found it.
	// (x, y)
    public void communicate() {
        //Message message = new Message("","","");
        //this.getEnvironment().receiveMessage(message); // this will send the message to the broadcast channel of the environment
    	
    	/**
    	 *  send messages 
    	 */
    	// send fuel station
    	
//    	if (this.memory.getFuelStation() != null) {
//            MyMessage myMessage = new MyMessage(this.getName(), "ALL", "FuelStation", this.memory.getFuelStation());
//        	this.getEnvironment().receiveMessage(myMessage);
//    	}
//    	
//    	// sensedObjects in a grid, range: 3 * 3
//    	int shareRange = Parameters.defaultSensorRange;
//    	Bag sensedObjects = new Bag();
//        IntBag objectXCoords = new IntBag();
//        IntBag objectYCoords = new IntBag();
//        
//        this.memory.getMemoryGrid().getNeighborsMaxDistance(x, y, shareRange, false, sensedObjects, objectXCoords, objectYCoords);
//		assert (sensedObjects.size() == objectXCoords.size() && sensedObjects.size() == objectYCoords.size());
//
//        MyMessage myMessage = new MyMessage(this.getName(), "ALL", "GRID", sensedObjects, new Int2D(x, y));
//    	this.getEnvironment().receiveMessage(myMessage);
    	
//  		Message msgFuel = new Message("", "");
//  		Int2D fuel = new Int2D(this.memory.getFuelStation().getX(), this.memory.getFuelStation().getY());
//  		msgFuel.addFuelStationPosition(fuel);
//  		this.getEnvironment().receiveMessage(msgFuel);
//    	

 		Message msg = new Message(this.getName(), "");
    	Bag sensedObjects = new Bag();
    	IntBag objectXCoords = new IntBag();
    	IntBag objectYCoords = new IntBag();
    	this.memory.getMemoryGrid().getNeighborsMaxDistance(x, y, Parameters.defaultSensorRange, false, sensedObjects, objectXCoords, objectYCoords);

  		msg.addFuelStationPosition(this.memory.getFuelStation());
  		msg.addSensedObjects(sensedObjects, new Int2D(x, y));
  		this.getEnvironment().receiveMessage(msg);

    	/**
    	 *  get messages 
    	 */ 
    	
        //LinkedList<TWEntity> goals = new LinkedList<TWEntity>();
    }
   
    @Override
	protected TWThought think() {
		// receive messages and update memory
    	
    	
		ArrayList<Message> messages = this.getEnvironment().getMessages();
		for (Message m : messages) {
  			if (m == null || m.message == null || m.getFrom() == this.getName()) continue;
  			Int2D fuelStationPos = m.getFuelStationPosition();
  			if (this.memory.getFuelStation() == null && fuelStationPos != null) {
  				this.memory.setFuelStation(fuelStationPos.x, fuelStationPos.y);
//  				break;
  			}
  			Bag sharedObjects = m.getSensedObjects();
  			Int2D posAgent = m.getAgentPosition();
  			if (sharedObjects != null) {
  				this.memory.mergeMemory(sharedObjects, posAgent);
  			}
  		}
			  
		
//    	for (int i = 0; i < messages.size(); i++) {
//    		Message myMessage = (Message) messages.get(i);
        	// fuel station
//    		if (myMessage1.getFrom() != this.getName() && myMessage1.getMessage() == "FuelStation") {
//    			this.memory.setFuelStatin(myMessage1.getFuelStation().x, myMessage1.getFuelStation().y);;
//    		} 
//    		// merge memory
//    		else if(myMessage1.getFrom() != this.getName() && myMessage1.getMessage() == "GRID") {
//    			Bag sharedObjects = myMessage1.getSensedObjects();
//    			Int2D posAgent = myMessage1.getPosAgent();
//    			this.memory.mergeMemory(sharedObjects, posAgent);
//    		}
//    		if (this.memory.getFuelStation() == null) {
//    			if (myMessage.getFrom() != this.getName() && myMessage.getFuelStationPosition() != null) {
//        			this.memory.setFuelStation(myMessage.getFuelStationPosition().x, myMessage.getFuelStationPosition().y);
//    		
    	
		// TODO Auto-generated method stub
		//getMemory().getClosestObjectInSensorRange(Tile.class);
//		neighborTiles = this.memory.getNearbyObject(x, y, 50, TWTile.class);
//		neighborHoles = this.memory.getNearbyHole(x, y, fuelThreshold);
		
//		neighborTiles = this.memory.getNearbyObjects(x, y, 50, TWTile.class);
//		neighborHoles = this.memory.getNearbyObjects(x, y, 50, TWHole.class);
		
//		TWTile tileTarget = (TWTile) detectObjectNoCollision(neighborTiles, this.memory.neighbouringAgents);
//		TWHole holeTarget = (TWHole) detectObjectNoCollision(neighborHoles, this.memory.neighbouringAgents);
		
		TWTile tileTarget = this.getMemory().getNearbyTile(x, y, 50);
		TWHole holeTarget = this.getMemory().getNearbyHole(x, y, 50);
		
		
		/* Decide the mode based on the current info of agent 
		 * 
		 * Default: INSPECT
		 * found fuel staion and fuel is low: REFUEL
		 * fuel is below a defined level: WAIT
		 * */
		
		mode = Mode.INSPECT;
		
		// if low fuel || slightly higher than limit, but cannot get to fuel station 
		if (memory.getFuelStation() == null) {
			mode = Mode.FIND_FUEL_STATION;
		} 
		else if (memory.getFuelStation() != null 
				&& (this.getFuelLevel() < this.fuelThreshold 
						|| this.getFuelLevel() * 0.9 < this.getDistanceTo(memory.getFuelStation().x, memory.getFuelStation().y))) {
			mode = Mode.REFUEL;
		} 
		else if (this.getFuelLevel() < this.fuelThreshold){
			mode = Mode.WAIT;
			return new TWThought(TWAction.MOVE, TWDirection.Z);
		} 
//		else if (this.hasTile() && holeTarget != null) {
//			mode = Mode.FILL;
//		} else if (!this.hasTile() && tileTarget != null) {
//			mode = Mode.COLLECT;
//		} 


		// current location of agent, get the object.
		Object curLocObj = this.memory.getMemoryGrid().get(x, y);
		if (curLocObj != null) {
//			System.out.println("curLocObj: " + curLocObj.getClass());
		}
		if (curLocObj instanceof TWFuelStation &&
				 (this.fuelLevel < (0.75 * Parameters.defaultFuelLevel))) {
			planner.getGoals().add(0, new Int2D(this.x, this.y));
			return new TWThought(TWAction.REFUEL, null);
		} else if (curLocObj instanceof TWHole && 
				this.getEnvironment().canPutdownTile((TWHole)curLocObj, this) &&
				this.hasTile()) 
		{
				planner.getGoals().add(0, new Int2D(this.x, this.y));
				return new TWThought(TWAction.PUTDOWN, null);
	
		} else if (curLocObj instanceof TWTile &&
				this.getEnvironment().canPickupTile((TWTile) curLocObj, this) &&
				this.carriedTiles.size() < 3) 
		{
				planner.getGoals().add(0, new Int2D(this.x, this.y));
				return new TWThought(TWAction.PICKUP, null);
		} 
		
		
		// Mode Goals Thought
		if (mode == Mode.FIND_FUEL_STATION) {
//			planner.voidGoals();
//			planner.voidPlan();
			if (this.planner.getGoals().isEmpty()) {
				// add goals
				addGoalsForFuelStation();
			}
			else {
				if (planner.getGoals().contains(new Int2D(this.x, this.y))) {
					int index = planner.getGoals().indexOf(new Int2D(this.x, this.y));
					if (index != -1) planner.getGoals().remove(index);
				}
			}	
		}
		
		else if (mode == Mode.INSPECT){
//			System.out.println(this.name + ", index =  " + this.index);

			if (this.planner.getGoals().size() > 20) {
				planner.voidGoals();
				planner.voidPlan();
//				throw new UnsupportedOperationException("too much goals!!");

			}
			if (this.planner.getGoals().isEmpty()) {
				// add goals
				addGoalsForInspector();
				if (this.index == 1) {
//					System.out.println(this.name + ", index =  " + this.index);
					Collections.reverse(this.planner.getGoals()); 
				}
				// System.out.println(this.name + ": Begin to inspect, I am added inspecting goals!!!!!!! " );
			}
			if (!planner.getGoals().isEmpty()) {
				// System.out.println(this.name + ": I have goals, and I am inspecting !!!!!!! " );
				if (planner.getGoals().contains(new Int2D(this.x, this.y))) {
					// System.out.println(this.name + ": I reached one subgoal, and I am changing my plan ahead to next goal!!!!!!! " );
					int temp = planner.getGoals().indexOf(new Int2D(this.x, this.y));
					if (index != -1) planner.getGoals().remove(temp);
				}			
			}

		}
		else {
			planner.voidGoals();
			planner.voidPlan();
			if (mode == Mode.REFUEL) {
				planner.getGoals().add(memory.getFuelStation());
			}  else if ((mode == Mode.FILL)){
				planner.getGoals().add(new Int2D(holeTarget.getX(), holeTarget.getY()));
			} else if ((mode == Mode.COLLECT)){
				planner.getGoals().add(new Int2D(tileTarget.getX(), tileTarget.getY()));
			} else if (mode == Mode.WAIT){
				return new TWThought(TWAction.MOVE, TWDirection.Z);
			} else if (mode == Mode.EXPLORE) {
				return RandomMoveThought();
			} 
		}
//		System.out.println(this.getName());
		for (int i = 0; i < planner.getGoals().size(); i++) {
//			System.out.println("Goals " + i + ": " + planner.getGoals().get(i));			
		}	
		if (this.planner.getGoals().isEmpty()){
			return RandomMoveThought();
		}
		
		planner.generatePlan();
		
		// if it is an obstacle isCellBlocked, cannot reach the goal
		if (!planner.hasPlan()) {
			if (this.mode == Mode.FIND_FUEL_STATION) {
				Int2D newGoal = generateRandomNearCell(planner.getGoals().get(0));
				planner.getGoals().set(0, newGoal);
				planner.generatePlan();
			} else {
				return new TWThought(TWAction.MOVE, TWDirection.Z);
			}
		}
		if (!planner.hasPlan()) {
			return RandomMoveThought();
		} 
		
		TWDirection dir = planner.execute();
		return new TWThought(TWAction.MOVE, dir);
//		} else if (mode == Mode.FILL) {
//			// 获取neighbour_hole, add to goal
//		} else if (mode == Mode.COLLECT) {
//			// 获取neighbour_tile, add to goal
//		} 

	}
	
private Int2D generateRandomNearCell(Int2D goalPos) {
		// TODO Auto-generated method stub
	  ArrayList<Int2D> dirs = new ArrayList<Int2D>();
	  int x = goalPos.getX();
	  int y = goalPos.getY();
	  
	  if ((y-1) >= 0 
			  && !this.memory.isCellBlocked(x, y-1)) {
		  dirs.add(new Int2D(x, y-1));
	  }
	  if ((y+1) < this.getEnvironment().getyDimension() 
			  && !this.memory.isCellBlocked(x, y+1)) {
		  dirs.add(new Int2D(x, y+1));
	  }
	  if ((x+1) < this.getEnvironment().getxDimension() 
			  && !this.memory.isCellBlocked(x+1, y)) {
		  dirs.add(new Int2D(x+1, y));
	  }
	  if ((x-1) >= 0 
			  && !this.memory.isCellBlocked(x-1, y)) {
		  dirs.add(new Int2D(x-1, y));
	  }
	  
	  if (dirs.size() > 0) {
		   int random_num = this.getEnvironment().random.nextInt(dirs.size());
		   return dirs.get(random_num);
	  }
	  else {
		  System.out.println("No where to go!");
		  return null;
	  }
	}


private Int2D getPositionAdd(Int2D base, Int2D position) {
		// TODO Auto-generated method stub
		return new Int2D (base.x + position.x, base.y + position.y) ;
	}
public void addGoalsForFuelStation() {
	planner.voidGoals();
	planner.voidPlan();
	
	Int2D[] basePos = {new Int2D(0, 0), 
			new Int2D(Parameters.xDimension/2, 0), 
			new Int2D(0, Parameters.yDimension/2), 
			new Int2D(Parameters.xDimension/2, Parameters.yDimension/2)};

	Int2D base = basePos[this.index];
	Int2D position = new Int2D (Parameters.defaultSensorRange, Parameters.defaultSensorRange);
	int depth = Parameters.defaultSensorRange;

	while(depth <= Parameters.xDimension/2) {
		// point 0 / 4
		int posX = position.x;
		int posY = position.y;
		planner.getGoals().add(getPositionAdd(base, position));
		
		// point 1
		posX = Parameters.xDimension/2 - Parameters.defaultSensorRange-1;
		position = new Int2D (posX, position.y);
		planner.getGoals().add(getPositionAdd(base, position));
		
		// point 2
		posY += Parameters.defaultSensorRange * 2 + 1;
		depth = depth + Parameters.defaultSensorRange * 2 + 1;
		if (depth > Parameters.xDimension/2) {
			break;
		}
		position = new Int2D (position.x, posY);
		planner.getGoals().add(getPositionAdd(base, position));
		
		// point 3
		posX = Parameters.defaultSensorRange;
		position = new Int2D (posX, position.y);
		planner.getGoals().add(getPositionAdd(base, position));
		depth = depth + (Parameters.defaultSensorRange * 2 + 1);
		
		// point 4
		posY += Parameters.defaultSensorRange * 2 + 1;
		position = new Int2D (position.x, posY);
	}
}

public void addGoalsForInspector() {
	Int2D position = new Int2D (Parameters.defaultSensorRange, Parameters.defaultSensorRange);
	int depth = Parameters.defaultSensorRange;
	
	while(depth <= Parameters.xDimension) {
		// point 0 / 4
		int posX = position.x;
		int posY = position.y;
		planner.getGoals().add(position);
		
		// point 1
		posX = Parameters.xDimension - Parameters.defaultSensorRange - 1;
		
		if (this.getEnvironment().isInBounds(posX, posY)) {
			position = new Int2D (posX, position.y);
			planner.getGoals().add(position);
		}
		
		// point 2
		posY += Parameters.defaultSensorRange * 2 + 1;
		depth = depth + Parameters.defaultSensorRange * 2 + 1;
		if (depth > Parameters.xDimension) {
			break;
		}
		position = new Int2D (position.x, posY);
		planner.getGoals().add(position);
		
		// point 3
		posX = Parameters.defaultSensorRange;;
		position = new Int2D (posX, position.y);
		planner.getGoals().add(position);
		depth = depth + (Parameters.defaultSensorRange * 2 + 1);
		
		// point 4
		posY += Parameters.defaultSensorRange * 2 + 1;
		position = new Int2D (position.x, posY);
	}
}


//	protected Object getClosest() {
//		TWTile tile = (TWTile) this.memory.closestInSensorRange.get(TWTile);
//		TWHole hole = (TWHole) this.memory.closestInSensorRange.get(TWHole);
//		return 
//	}

	private TWEntity detectObjectNoCollision(PriorityQueue<TWEntity> neighborObjects, List<TWAgent> neighbouringAgents) {
		// TODO Auto-generated method stub
		LinkedList<TWEntity> tiles = null;
		while(!neighborObjects.isEmpty()) {
			TWEntity tile = neighborObjects.poll();
			for (int i = 0; i < neighbouringAgents.size(); i++) {
				// if dis < agentDis && tile is not agent closest
//				if (this.getDistanceTo(tile) < neighbouringAgents.get(i).getDistanceTo(tile)) {
//					TWEntity closestHole = neighbouringAgents.get(i).memory.getClosestObjectInSensorRange(tile.getClass());
//					if (!tile.equals(closestHole)) {
//						return tile; 
//					}
//				}					
				if (this.getDistanceTo(tile) >= neighbouringAgents.get(i).getDistanceTo(tile)) {
					continue;
				}
			}
			return tile;
//			tiles.add(tile);
		}
		return null;
	}

	@Override
	protected void act(TWThought thought) {
		// TODO Auto-generated method stub
		//You can do:
        //move(thought.getDirection())
        //pickUpTile(Tile)
        //putTileInHole(Hole)
        //refuel()
		Int2D curGoal = planner.getCurrentGoal();
		
		try {
			switch (thought.getAction()) {
			case MOVE:
//				System.out.println("Direction:" + thought.getDirection());
				move(thought.getDirection());
				break;
			case PICKUP:
				TWTile tile = (TWTile) memory.getMemoryGrid().get(this.x, this.y);
				pickUpTile(tile);
//				planner.getGoals().clear();
				break;
			case PUTDOWN:
				TWHole hole = (TWHole) memory.getMemoryGrid().get(this.x, this.y);
				putTileInHole(hole);
//				planner.getGoals().clear();
				break;
			case REFUEL:
				refuel();
//				planner.getGoals().clear();
				break;
			}
//            this.move(thought.getDirection());
            
        } catch (CellBlockedException ex) {
//        	System.out.println("Current mode: "+this.mode);
    		System.out.println("Size of goal: "+this.planner.getGoals().size());
    		System.out.println("N: " + this.memory.isCellBlocked(x, y-1));
    		System.out.println("S: " + this.memory.isCellBlocked(x, y+1));
    		System.out.println("E: " + this.memory.isCellBlocked(x+1, y));
    		System.out.println("W: " + this.memory.isCellBlocked(x-1, y));
			System.out.println("Cell is blocked. Current Position: " + Integer.toString(this.x) + ", " + Integer.toString(this.y));
        }
//		System.out.println("Step " + this.getEnvironment().schedule.getSteps());
//		System.out.println(name + " score: " + this.score);
////		System.out.println("Assigned Zone: " + Integer.toString(agentZones[agentIdx]));
////		System.out.println("Mode: " + mode.name());
//		System.out.println("Position: " + Integer.toString(this.x) + ", " + Integer.toString(this.y));
//		System.out.println("Current Mode: " + this.mode);
//		System.out.println("Size of goal: "+this.planner.getGoals().size());
//		
//		if (curGoal != null) {
//			System.out.println("Goal: " + curGoal.x + ", " + curGoal.y);
//		}
//		else
//			System.out.println("Goal: Nothing");
//		System.out.println("Tiles: " + this.carriedTiles.size());
//		System.out.println("Fuel Level: " + this.fuelLevel);
//		System.out.println("Fuel Station: " + this.memory.getFuelStation());
//		System.out.println("");

	}
	private TWThought RandomMoveThought() {
	  ArrayList<TWDirection> dirs = new ArrayList<TWDirection>();
	  int x = this.getX();
	  int y = this.getY();
	  
	  if ((y-1) >= 0 
			  && !this.memory.isCellBlocked(x, y-1)) {
		  dirs.add(TWDirection.N);
	  }
	  if ((y+1) < this.getEnvironment().getyDimension() 
			  && !this.memory.isCellBlocked(x, y+1)) {
		  dirs.add(TWDirection.S);
	  }
	  if ((x+1) < this.getEnvironment().getxDimension() 
			  && !this.memory.isCellBlocked(x+1, y)) {
		  dirs.add(TWDirection.E);
	  }
	  if ((x-1) >= 0 
			  && !this.memory.isCellBlocked(x-1, y)) {
		  dirs.add(TWDirection.W);
	  }
	  
	  if (dirs.size() > 0) {
	   int random_num = this.getEnvironment().random.nextInt(dirs.size());
	   return new TWThought(TWAction.MOVE, dirs.get(random_num));
	  }
	  else {
	   System.out.println("No where to go!");
	   return new TWThought(TWAction.MOVE, TWDirection.Z);
	  }
    }
	
//	MyMemory
	@Override  
	public TWAgentWorkingMemory getMemory() {
        return this.memory;
    }
	@Override
	public String getName() {
		// TODO Auto-generated method stub
		return name;
	}
		
	

}





//package tileworld.agent;
//
//import java.awt.Color;
//import java.util.ArrayList;
//import java.util.Collections;
//import java.util.LinkedList;
//import java.util.List;
//import java.util.PriorityQueue;
//
//import sim.display.GUIState;
//import sim.portrayal.Inspector;
//import sim.portrayal.LocationWrapper;
//import sim.portrayal.Portrayal;
//import sim.util.Bag;
//import sim.util.Int2D;
//import sim.util.IntBag;
//import tileworld.Parameters;
//import tileworld.agent.MyAgent.Mode;
//import tileworld.environment.TWDirection;
//import tileworld.environment.TWEntity;
//import tileworld.environment.TWEnvironment;
//import tileworld.environment.TWFuelStation;
//import tileworld.environment.TWHole;
//import tileworld.environment.TWTile;
//import tileworld.exceptions.CellBlockedException;
//import tileworld.planners.DefaultTWPlanner;
//
///*
// * use createAgent(new MyInspectorAgent(zoneIndentity, "agentInspector 0", pos.getX(), pos.getY(), this, Parameters.defaultFuelLevel));
// * 	zoneIndentity range: [0,1,2,3]
// * */
//
//public class MyInspectorAgent extends TWAgent {
//	
//	/* 
//	 * EXPLORE: random walking, when no Tiles observed in the agents memory
//	 * COLLECT: fuel is adequate, tiles < 3, have position 
//	 * FILL: tiles = 3, have 
//	 * REFUEL: fuel below a threshold, already know the fuel station
//	 * WAIT: fuel below a threshold, wait other agent found fuel station 
//	 * */
//	enum Mode {
//		EXPLORE, COLLECT, FILL, REFUEL, WAIT, FIND_FUEL_STATION, INSPECT
//	}
//	
//	private int index;
//	private String name;
//	private MyMemory memory;
//	private DefaultTWPlanner planner;
//	private Mode mode;
//	private double fuelThreshold;
//	
//	TWEntity neighborTile;
//	TWEntity neighborHole;
//	
//	PriorityQueue<TWEntity> neighborTiles;
//	PriorityQueue<TWEntity> neighborHoles;
//	
//	
//	public MyInspectorAgent(int index, String name, int xpos, int ypos, TWEnvironment env, double fuelLevel) {
//        super(xpos,ypos,env,fuelLevel);
//        this.index = index;
//        this.name = name;
//        this.fuelThreshold = this.getFuelLevel() / 10 ;
//        this.planner = new DefaultTWPlanner(this);
//		this.sensor = new TWAgentSensor(this, Parameters.defaultSensorRange);
//        this.memory = new MyMemory(this, env.schedule, env.getxDimension(), env.getyDimension());
////		this.memory = (MyMemory) this.memory;
//	}
//	
//	
//	public MyInspectorAgent(int xpos, int ypos, TWEnvironment env, double fuelLevel) {
//		super(xpos, ypos, env, fuelLevel);
//		this.planner = new DefaultTWPlanner(this);
//	}
//	
//	
//	// get position FuelStation, and spread its if this agent found it.
//	// (x, y)
//    public void communicate() {
//        //Message message = new Message("","","");
//        //this.getEnvironment().receiveMessage(message); // this will send the message to the broadcast channel of the environment
//    	
//    	/**
//    	 *  send messages 
//    	 */
//    	// send fuel station
//    	
////    	if (this.memory.getFuelStation() != null) {
////            MyMessage myMessage = new MyMessage(this.getName(), "ALL", "FuelStation", this.memory.getFuelStation());
////        	this.getEnvironment().receiveMessage(myMessage);
////    	}
////    	
////    	// sensedObjects in a grid, range: 3 * 3
////    	int shareRange = Parameters.defaultSensorRange;
////    	Bag sensedObjects = new Bag();
////        IntBag objectXCoords = new IntBag();
////        IntBag objectYCoords = new IntBag();
////        
////        this.memory.getMemoryGrid().getNeighborsMaxDistance(x, y, shareRange, false, sensedObjects, objectXCoords, objectYCoords);
////		assert (sensedObjects.size() == objectXCoords.size() && sensedObjects.size() == objectYCoords.size());
////
////        MyMessage myMessage = new MyMessage(this.getName(), "ALL", "GRID", sensedObjects, new Int2D(x, y));
////    	this.getEnvironment().receiveMessage(myMessage);
//    	
////  		Message msgFuel = new Message("", "");
////  		Int2D fuel = new Int2D(this.memory.getFuelStation().getX(), this.memory.getFuelStation().getY());
////  		msgFuel.addFuelStationPosition(fuel);
////  		this.getEnvironment().receiveMessage(msgFuel);
////    	
//
// 		Message msg = new Message(this.getName(), "");
//    	Bag sensedObjects = new Bag();
//    	IntBag objectXCoords = new IntBag();
//    	IntBag objectYCoords = new IntBag();
//    	this.memory.getMemoryGrid().getNeighborsMaxDistance(x, y, Parameters.defaultSensorRange, false, sensedObjects, objectXCoords, objectYCoords);
//
//  		msg.addFuelStationPosition(this.memory.getFuelStation());
//  		msg.addSensedObjects(sensedObjects, new Int2D(x, y));
//  		this.getEnvironment().receiveMessage(msg);
//
//    	/**
//    	 *  get messages 
//    	 */ 
//    	
//        //LinkedList<TWEntity> goals = new LinkedList<TWEntity>();
//    }
//   
//    @Override
//	protected TWThought think() {
//		// receive messages and update memory
//    	
//    	
//		ArrayList<Message> messages = this.getEnvironment().getMessages();
//		for (Message m : messages) {
//  			if (m == null || m.message == null || m.getFrom() == this.getName()) continue;
//  			Int2D fuelStationPos = m.getFuelStationPosition();
//  			if (this.memory.getFuelStation() == null && fuelStationPos != null) {
//  				this.memory.setFuelStation(fuelStationPos.x, fuelStationPos.y);
////  				break;
//  			}
//  			Bag sharedObjects = m.getSensedObjects();
//  			Int2D posAgent = m.getAgentPosition();
//  			if (sharedObjects != null) {
//  				this.memory.mergeMemory(sharedObjects, posAgent);
//  			}
//  		}
//			  
//		
////    	for (int i = 0; i < messages.size(); i++) {
////    		Message myMessage = (Message) messages.get(i);
//        	// fuel station
////    		if (myMessage1.getFrom() != this.getName() && myMessage1.getMessage() == "FuelStation") {
////    			this.memory.setFuelStatin(myMessage1.getFuelStation().x, myMessage1.getFuelStation().y);;
////    		} 
////    		// merge memory
////    		else if(myMessage1.getFrom() != this.getName() && myMessage1.getMessage() == "GRID") {
////    			Bag sharedObjects = myMessage1.getSensedObjects();
////    			Int2D posAgent = myMessage1.getPosAgent();
////    			this.memory.mergeMemory(sharedObjects, posAgent);
////    		}
////    		if (this.memory.getFuelStation() == null) {
////    			if (myMessage.getFrom() != this.getName() && myMessage.getFuelStationPosition() != null) {
////        			this.memory.setFuelStation(myMessage.getFuelStationPosition().x, myMessage.getFuelStationPosition().y);
////    		
//    	
//		// TODO Auto-generated method stub
//		//getMemory().getClosestObjectInSensorRange(Tile.class);
////		neighborTiles = this.memory.getNearbyObject(x, y, 50, TWTile.class);
////		neighborHoles = this.memory.getNearbyHole(x, y, fuelThreshold);
//		
////		neighborTiles = this.memory.getNearbyObjects(x, y, 50, TWTile.class);
////		neighborHoles = this.memory.getNearbyObjects(x, y, 50, TWHole.class);
//		
////		TWTile tileTarget = (TWTile) detectObjectNoCollision(neighborTiles, this.memory.neighbouringAgents);
////		TWHole holeTarget = (TWHole) detectObjectNoCollision(neighborHoles, this.memory.neighbouringAgents);
//		
//		TWTile tileTarget = this.getMemory().getNearbyTile(x, y, 50);
//		TWHole holeTarget = this.getMemory().getNearbyHole(x, y, 50);
//		
//		
//		/* Decide the mode based on the current info of agent 
//		 * 
//		 * Default: INSPECT
//		 * found fuel staion and fuel is low: REFUEL
//		 * fuel is below a defined level: WAIT
//		 * */
//		
//		mode = Mode.INSPECT;
//		
//		// if low fuel || slightly higher than limit, but cannot get to fuel station 
//		if (memory.getFuelStation() == null) {
//			mode = Mode.FIND_FUEL_STATION;
//		} 
//		else if (memory.getFuelStation() != null 
//				&& (this.getFuelLevel() < this.fuelThreshold 
//						|| this.getFuelLevel() * 0.9 < this.getDistanceTo(memory.getFuelStation().x, memory.getFuelStation().y))) {
//			mode = Mode.REFUEL;
//		} 
//		else if (this.getFuelLevel() < this.fuelThreshold){
//			mode = Mode.WAIT;
//			return new TWThought(TWAction.MOVE, TWDirection.Z);
//		} 
////		else if (this.hasTile() && holeTarget != null) {
////			mode = Mode.FILL;
////		} else if (!this.hasTile() && tileTarget != null) {
////			mode = Mode.COLLECT;
////		} 
//
//		// current location of agent, get the object.
//		Object curLocObj = this.memory.getMemoryGrid().get(x, y);
//		if (curLocObj != null) {
////			System.out.println("curLocObj: " + curLocObj.getClass());
//		}
//		if (curLocObj instanceof TWFuelStation &&
//				 (this.fuelLevel < (0.75 * Parameters.defaultFuelLevel))) {
//			planner.getGoals().add(0, new Int2D(this.x, this.y));
//			return new TWThought(TWAction.REFUEL, null);
//		} else if (curLocObj instanceof TWHole && 
//				this.getEnvironment().canPutdownTile((TWHole)curLocObj, this) &&
//				this.hasTile()) 
//		{
//				planner.getGoals().add(0, new Int2D(this.x, this.y));
//				return new TWThought(TWAction.PUTDOWN, null);
//	
//		} else if (curLocObj instanceof TWTile &&
//				this.getEnvironment().canPickupTile((TWTile) curLocObj, this) &&
//				this.carriedTiles.size() < 3) 
//		{
//				planner.getGoals().add(0, new Int2D(this.x, this.y));
//				return new TWThought(TWAction.PICKUP, null);
//		} 
//		
//		
//		// Mode Goals Thought
//		if (mode == Mode.FIND_FUEL_STATION) {
////			planner.voidGoals();
////			planner.voidPlan();
//			if (this.planner.getGoals().isEmpty()) {
//				// add goals
//				addGoalsForFuelStation();
//			}
//			else {
//				if (planner.getGoals().contains(new Int2D(this.x, this.y))) {
//					int index = planner.getGoals().indexOf(new Int2D(this.x, this.y));
//					if (index != -1) planner.getGoals().remove(index);
//				}
//			}	
//		}
//		
//		else if (mode == Mode.INSPECT){
////			System.out.println(this.name + ", index =  " + this.index);
//
//			if (this.planner.getGoals().size() > 20) {
//				planner.voidGoals();
//				planner.voidPlan();
////				throw new UnsupportedOperationException("too much goals!!");
//
//			}
//			if (this.planner.getGoals().isEmpty()) {
//				// add goals
//				addGoalsForInspector();
//				if (this.index == 1) {
////					System.out.println(this.name + ", index =  " + this.index);
//					Collections.reverse(this.planner.getGoals()); 
//				}
//				// System.out.println(this.name + ": Begin to inspect, I am added inspecting goals!!!!!!! " );
//			}
//			if (!planner.getGoals().isEmpty()) {
//				// System.out.println(this.name + ": I have goals, and I am inspecting !!!!!!! " );
//				if (planner.getGoals().contains(new Int2D(this.x, this.y))) {
//					// System.out.println(this.name + ": I reached one subgoal, and I am changing my plan ahead to next goal!!!!!!! " );
//					int temp = planner.getGoals().indexOf(new Int2D(this.x, this.y));
//					if (index != -1) planner.getGoals().remove(temp);
//				}			
//			}
//
//		}
//		else {
//			planner.voidGoals();
//			planner.voidPlan();
//			if (mode == Mode.REFUEL) {
//				planner.getGoals().add(memory.getFuelStation());
//			}  else if ((mode == Mode.FILL)){
//				planner.getGoals().add(new Int2D(holeTarget.getX(), holeTarget.getY()));
//			} else if ((mode == Mode.COLLECT)){
//				planner.getGoals().add(new Int2D(tileTarget.getX(), tileTarget.getY()));
//			} else if (mode == Mode.WAIT){
//				return new TWThought(TWAction.MOVE, TWDirection.Z);
//			} else if (mode == Mode.EXPLORE) {
//				return RandomMoveThought();
//			} 
//		}
////		System.out.println(this.getName());
//		for (int i = 0; i < planner.getGoals().size(); i++) {
////			System.out.println("Goals " + i + ": " + planner.getGoals().get(i));			
//		}	
//		if (this.planner.getGoals().isEmpty()){
//			return RandomMoveThought();
//		}
//		
//		planner.generatePlan();
//		
//		// if it is an obstacle isCellBlocked, cannot reach the goal
//		if (!planner.hasPlan()) {
//			if (this.mode == Mode.FIND_FUEL_STATION) {
//				Int2D newGoal = generateRandomNearCell(planner.getGoals().get(0));
//				planner.getGoals().set(0, newGoal);
//				planner.generatePlan();
//			} else {
//				return new TWThought(TWAction.MOVE, TWDirection.Z);
//			}
//		}
//		if (!planner.hasPlan()) {
//			return RandomMoveThought();
//		} 
//		
//		TWDirection dir = planner.execute();
//		return new TWThought(TWAction.MOVE, dir);
////		} else if (mode == Mode.FILL) {
////			// neighbour_hole, add to goal
////		} else if (mode == Mode.COLLECT) {
////			// neighbour_tile, add to goal
////		} 
//
//	}
//	
//private Int2D generateRandomNearCell(Int2D goalPos) {
//		// TODO Auto-generated method stub
//	  ArrayList<Int2D> dirs = new ArrayList<Int2D>();
//	  int x = goalPos.getX();
//	  int y = goalPos.getY();
//	  
//	  if ((y-1) >= 0 
//			  && !this.memory.isCellBlocked(x, y-1)) {
//		  dirs.add(new Int2D(x, y-1));
//	  }
//	  if ((y+1) < this.getEnvironment().getyDimension() 
//			  && !this.memory.isCellBlocked(x, y+1)) {
//		  dirs.add(new Int2D(x, y+1));
//	  }
//	  if ((x+1) < this.getEnvironment().getxDimension() 
//			  && !this.memory.isCellBlocked(x+1, y)) {
//		  dirs.add(new Int2D(x+1, y));
//	  }
//	  if ((x-1) >= 0 
//			  && !this.memory.isCellBlocked(x-1, y)) {
//		  dirs.add(new Int2D(x-1, y));
//	  }
//	  
//	  if (dirs.size() > 0) {
//		   int random_num = this.getEnvironment().random.nextInt(dirs.size());
//		   return dirs.get(random_num);
//	  }
//	  else {
//		  System.out.println("No where to go!");
//		  return null;
//	  }
//	}
//
//
//private Int2D getPositionAdd(Int2D base, Int2D position) {
//		// TODO Auto-generated method stub
//		return new Int2D (base.x + position.x, base.y + position.y) ;
//	}
//public void addGoalsForFuelStation() {
//	planner.voidGoals();
//	planner.voidPlan();
//	
//	Int2D[] basePos = {new Int2D(0, 0), 
//			new Int2D(Parameters.xDimension/2, 0), 
//			new Int2D(0, Parameters.yDimension/2), 
//			new Int2D(Parameters.xDimension/2, Parameters.yDimension/2)};
//
//	Int2D base = basePos[this.index];
//	Int2D position = new Int2D (Parameters.defaultSensorRange, Parameters.defaultSensorRange);
//	int depth = Parameters.defaultSensorRange;
//
//	while(depth <= Parameters.xDimension/2) {
//		// point 0 / 4
//		int posX = position.x;
//		int posY = position.y;
//		planner.getGoals().add(getPositionAdd(base, position));
//		
//		// point 1
//		posX = Parameters.xDimension/2 - Parameters.defaultSensorRange-1;
//		position = new Int2D (posX, position.y);
//		planner.getGoals().add(getPositionAdd(base, position));
//		
//		// point 2
//		posY += Parameters.defaultSensorRange * 2 + 1;
//		depth = depth + Parameters.defaultSensorRange * 2 + 1;
//		if (depth > Parameters.xDimension/2) {
//			break;
//		}
//		position = new Int2D (position.x, posY);
//		planner.getGoals().add(getPositionAdd(base, position));
//		
//		// point 3
//		posX = Parameters.defaultSensorRange;
//		position = new Int2D (posX, position.y);
//		planner.getGoals().add(getPositionAdd(base, position));
//		depth = depth + (Parameters.defaultSensorRange * 2 + 1);
//		
//		// point 4
//		posY += Parameters.defaultSensorRange * 2 + 1;
//		position = new Int2D (position.x, posY);
//	}
//}
//
//public void addGoalsForInspector() {
//	Int2D position = new Int2D (Parameters.defaultSensorRange, Parameters.defaultSensorRange);
//	int depth = Parameters.defaultSensorRange;
//	
//	while(depth <= Parameters.xDimension) {
//		// point 0 / 4
//		int posX = position.x;
//		int posY = position.y;
//		planner.getGoals().add(position);
//		
//		// point 1
//		posX = Parameters.xDimension - Parameters.defaultSensorRange - 1;
//		
//		if (this.getEnvironment().isInBounds(posX, posY)) {
//			position = new Int2D (posX, position.y);
//			planner.getGoals().add(position);
//		}
//		
//		// point 2
//		posY += Parameters.defaultSensorRange * 2 + 1;
//		depth = depth + Parameters.defaultSensorRange * 2 + 1;
//		if (depth > Parameters.xDimension) {
//			break;
//		}
//		position = new Int2D (position.x, posY);
//		planner.getGoals().add(position);
//		
//		// point 3
//		posX = Parameters.defaultSensorRange;;
//		position = new Int2D (posX, position.y);
//		planner.getGoals().add(position);
//		depth = depth + (Parameters.defaultSensorRange * 2 + 1);
//		
//		// point 4
//		posY += Parameters.defaultSensorRange * 2 + 1;
//		position = new Int2D (position.x, posY);
//	}
//}
//
//
////	protected Object getClosest() {
////		TWTile tile = (TWTile) this.memory.closestInSensorRange.get(TWTile);
////		TWHole hole = (TWHole) this.memory.closestInSensorRange.get(TWHole);
////		return 
////	}
//
//	private TWEntity detectObjectNoCollision(PriorityQueue<TWEntity> neighborObjects, List<TWAgent> neighbouringAgents) {
//		// TODO Auto-generated method stub
//		LinkedList<TWEntity> tiles = null;
//		while(!neighborObjects.isEmpty()) {
//			TWEntity tile = neighborObjects.poll();
//			for (int i = 0; i < neighbouringAgents.size(); i++) {
//				// if dis < agentDis && tile is not agent closest
////				if (this.getDistanceTo(tile) < neighbouringAgents.get(i).getDistanceTo(tile)) {
////					TWEntity closestHole = neighbouringAgents.get(i).memory.getClosestObjectInSensorRange(tile.getClass());
////					if (!tile.equals(closestHole)) {
////						return tile; 
////					}
////				}					
//				if (this.getDistanceTo(tile) >= neighbouringAgents.get(i).getDistanceTo(tile)) {
//					continue;
//				}
//			}
//			return tile;
////			tiles.add(tile);
//		}
//		return null;
//	}
//
//	@Override
//	protected void act(TWThought thought) {
//		// TODO Auto-generated method stub
//		//You can do:
//        //move(thought.getDirection())
//        //pickUpTile(Tile)
//        //putTileInHole(Hole)
//        //refuel()
//		Int2D curGoal = planner.getCurrentGoal();
//		
//		try {
//			switch (thought.getAction()) {
//			case MOVE:
////				System.out.println("Direction:" + thought.getDirection());
//				move(thought.getDirection());
//				break;
//			case PICKUP:
//				TWTile tile = (TWTile) memory.getMemoryGrid().get(this.x, this.y);
//				pickUpTile(tile);
////				planner.getGoals().clear();
//				break;
//			case PUTDOWN:
//				TWHole hole = (TWHole) memory.getMemoryGrid().get(this.x, this.y);
//				putTileInHole(hole);
////				planner.getGoals().clear();
//				break;
//			case REFUEL:
//				refuel();
////				planner.getGoals().clear();
//				break;
//			}
////            this.move(thought.getDirection());
//            
//        } catch (CellBlockedException ex) {
////        	System.out.println("Current mode: "+this.mode);
//    		System.out.println("Size of goal: "+this.planner.getGoals().size());
//
//    		System.out.println("N: " + this.memory.isCellBlocked(x, y-1));
//    		System.out.println("S: " + this.memory.isCellBlocked(x, y+1));
//    		System.out.println("E: " + this.memory.isCellBlocked(x+1, y));
//    		System.out.println("W: " + this.memory.isCellBlocked(x-1, y));
//			System.out.println("Cell is blocked. Current Position: " + Integer.toString(this.x) + ", " + Integer.toString(this.y));
//        }
////		System.out.println("Step " + this.getEnvironment().schedule.getSteps());
////		System.out.println(name + " score: " + this.score);
//////		System.out.println("Assigned Zone: " + Integer.toString(agentZones[agentIdx]));
//////		System.out.println("Mode: " + mode.name());
////		System.out.println("Position: " + Integer.toString(this.x) + ", " + Integer.toString(this.y));
////		System.out.println("Current Mode: " + this.mode);
////		System.out.println("Size of goal: "+this.planner.getGoals().size());
////		
////		if (curGoal != null) {
////			System.out.println("Goal: " + curGoal.x + ", " + curGoal.y);
////		}
////		else
////			System.out.println("Goal: Nothing");
////		System.out.println("Tiles: " + this.carriedTiles.size());
////		System.out.println("Fuel Level: " + this.fuelLevel);
////		System.out.println("Fuel Station: " + this.memory.getFuelStation());
////		System.out.println("");
//
//	}
//	private TWThought RandomMoveThought() {
//	  ArrayList<TWDirection> dirs = new ArrayList<TWDirection>();
//	  int x = this.getX();
//	  int y = this.getY();
//	  
//	  if ((y-1) >= 0 
//			  && !this.memory.isCellBlocked(x, y-1)) {
//		  dirs.add(TWDirection.N);
//	  }
//	  if ((y+1) < this.getEnvironment().getyDimension() 
//			  && !this.memory.isCellBlocked(x, y+1)) {
//		  dirs.add(TWDirection.S);
//	  }
//	  if ((x+1) < this.getEnvironment().getxDimension() 
//			  && !this.memory.isCellBlocked(x+1, y)) {
//		  dirs.add(TWDirection.E);
//	  }
//	  if ((x-1) >= 0 
//			  && !this.memory.isCellBlocked(x-1, y)) {
//		  dirs.add(TWDirection.W);
//	  }
//	  
//	  if (dirs.size() > 0) {
//	   int random_num = this.getEnvironment().random.nextInt(dirs.size());
//	   return new TWThought(TWAction.MOVE, dirs.get(random_num));
//	  }
//	  else {
//	   System.out.println("No where to go!");
//	   return new TWThought(TWAction.MOVE, TWDirection.Z);
//	  }
//    }
//	
////	MyMemory
//	@Override  
//	public TWAgentWorkingMemory getMemory() {
//        return this.memory;
//    }
//	@Override
//	public String getName() {
//		// TODO Auto-generated method stub
//		return name;
//	}
//		
//	
//
//}