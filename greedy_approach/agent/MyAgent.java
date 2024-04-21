package tileworld.agent;

import java.awt.Color;
import java.util.ArrayList;
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
import tileworld.agent.MyInspectorAgent.Mode;
import tileworld.environment.TWDirection;
import tileworld.environment.TWEntity;
import tileworld.environment.TWEnvironment;
import tileworld.environment.TWFuelStation;
import tileworld.environment.TWHole;
import tileworld.environment.TWObject;
import tileworld.environment.TWTile;
import tileworld.exceptions.CellBlockedException;
import tileworld.planners.DefaultTWPlanner;
//import tileworld.environment.TWFuelStation;


public class MyAgent extends TWAgent {
	
	/* 
	 * EXPLORE: random walking, when no Tiles observed in the agents memory
	 * COLLECT: fuel is adequate, tiles < 3, have position 
	 * FILL: tiles = 3, have 
	 * REFUEL: fuel below a threshold, already know the fuel station
	 * WAIT: fuel below a threshold, wait other agent found fuel station 
	 * */
	enum Mode {
		EXPLORE, COLLECT, FILL, REFUEL, WAIT, FIND_FUEL_STATION
	}

	private static final Object TWTile = null;

	private static final Object TWHole = null;
	
	private int index;
	private String name;
	private MyMemory memory;
	private DefaultTWPlanner planner;
	private Mode mode;
	private double fuelThreshold;
	
	TWObject neighborTile;
	TWObject neighborHole;
	
	public MyAgent(int index, String name, int xpos, int ypos, TWEnvironment env, double fuelLevel) {
        super(xpos,ypos,env,fuelLevel);
        this.index = index;
        this.name = name;
        this.fuelThreshold = this.getFuelLevel() / 10 ;
        this.planner = new DefaultTWPlanner(this);
		this.sensor = new TWAgentSensor(this, Parameters.defaultSensorRange);
        this.memory = new MyMemory(this, env.schedule, env.getxDimension(), env.getyDimension());
//		this.memory = (MyMemory) this.memory;
	}
	
	
//	public MyAgent(int xpos, int ypos, TWEnvironment env, double fuelLevel) {
//		super(xpos, ypos, env, fuelLevel);
//		this.planner = new DefaultTWPlanner(this);
//	}
	
	
	// get position FuelStation, and spread its if this agent found it.
	// (x, y)
    public void communicate() {
//        Message message = new Message("","","");
//        this.getEnvironment().receiveMessage(message); // this will send the message to the broadcast channel of the environment
//    	
//    	/**
//    	 *  send messages 
//    	 */
//    	// send fuel station
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
    	
    	
    	Message msg = new Message(this.getName(), "");
    	Bag sensedObjects = new Bag();
    	IntBag objectXCoords = new IntBag();
    	IntBag objectYCoords = new IntBag();
    	this.memory.getMemoryGrid().getNeighborsMaxDistance(x, y, Parameters.defaultSensorRange, false, sensedObjects, objectXCoords, objectYCoords);

  		msg.addFuelStationPosition(this.memory.getFuelStation());
  		msg.addSensedObjects(sensedObjects, new Int2D(x, y));
  		this.getEnvironment().receiveMessage(msg);
 	
    }
    
   
	@Override
	protected TWThought think() {
		/**
    	 *  get messages and update memory
    	 */ 
//		ArrayList<Message> messages = this.getEnvironment().getMessages();
//    	for (int i = 0; i < messages.size(); i++) {
//    		MyMessage myMessage1 = (MyMessage) messages.get(i);
//        	// fuel station
//    		if (myMessage1.getFrom() != this.getName() && myMessage1.getMessage() == "FuelStation") {
//    			this.memory.setFuelStation(myMessage1.getFuelStation().x, myMessage1.getFuelStation().y);;
//    		} 
//    		// merge memory
//    		else if(myMessage1.getFrom() != this.getName() && myMessage1.getMessage() == "GRID") {
//    			Bag sharedObjects = myMessage1.getSensedObjects();
//    			Int2D posAgent = myMessage1.getPosAgent();
//    			this.memory.mergeMemory(sharedObjects, posAgent);
//    		}
//    	} 
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
    	
    	
    	
		// TODO Auto-generated method stub
		//getMemory().getClosestObjectInSensorRange(Tile.class);
//		neighborTiles = this.memory.getNearbyObject(x, y, 50, TWTile.class);
//		neighborHoles = this.memory.getNearbyHole(x, y, fuelThreshold);
		
    	
    	PriorityQueue<TWEntity> neighborTiles;
    	PriorityQueue<TWEntity> neighborHoles;
    	
		neighborTiles = this.memory.getNearbyAllSortedObjects(x, y, 50, TWTile.class);
		neighborHoles = this.memory.getNearbyAllSortedObjects(x, y, 50, TWHole.class);
		
		
		ArrayList<TWEntity> tiles = detectObjectNoCollision(neighborTiles, this.memory.neighbouringAgents);
		ArrayList<TWEntity> holes = detectObjectNoCollision(neighborHoles, this.memory.neighbouringAgents);

		TWTile tileTarget = null;
		TWHole holeTarget = null;
		
		if (tiles != null) {
			tileTarget = (TWTile) tiles.get(0);
		}
		if (holes != null) {
			holeTarget = (TWHole) holes.get(0);
		}

		tileTarget = this.getMemory().getNearbyTile(x, y, 50);
		holeTarget = this.getMemory().getNearbyHole(x, y, 50);
		
		
		/* Decide the mode based on the current info of agent 
		 * 
		 * Default: EXPLORE
		 * found fuel staion and fuel is low: REFUEL
		 * fuel is below a defined level: WAIT
		 * no nearby tiles or holes, EXPLORE
		 * has tile on hand, FILL
		 * no tile on hand, COLLECT
		 * */
		mode = Mode.EXPLORE;
		
		// if low fuel || slightly higher than limit, but cannot get to fuel station 
		if (memory.getFuelStation() == null) {
			mode = Mode.FIND_FUEL_STATION;
		} else if (memory.getFuelStation() != null 
				&& (this.getFuelLevel() < this.fuelThreshold 
						|| this.getFuelLevel() * 0.9 < this.getDistanceTo(memory.getFuelStation().x, memory.getFuelStation().y))) {
			mode = Mode.REFUEL;
		} else if (this.getFuelLevel() < this.fuelThreshold){
			mode = Mode.WAIT;
			return new TWThought(TWAction.MOVE, TWDirection.Z);
		} else if (!(mode == Mode.FIND_FUEL_STATION)) {
			if (this.hasTile() && holeTarget != null) {
				mode = Mode.FILL;
			} else if (!this.hasTile() && tileTarget != null) {
				mode = Mode.COLLECT;
			} else mode = Mode.EXPLORE;
		} else mode = Mode.EXPLORE;

		/*
		 * fuel station: REFUELING
		 * pick up tile: COLLECTING
		 * fill a hole: FILLING
		 * */
		// current location of agent, get the object.
		Object curLocObj = this.memory.getMemoryGrid().get(x, y);
		if (curLocObj != null) {
//			System.out.println("curLocObj: " + curLocObj.getClass());
		}
		if (curLocObj instanceof TWFuelStation &&
				 (this.fuelLevel < (0.75 * Parameters.defaultFuelLevel))) {
//			planner.getGoals().add(new Int2D(this.x, this.y));
			return new TWThought(TWAction.REFUEL, null);
		} else if (curLocObj instanceof TWHole && 
				this.getEnvironment().canPutdownTile((TWHole)curLocObj, this) &&
				this.hasTile()) 
		{
//				planner.getGoals().add(new Int2D(this.x, this.y));
				return new TWThought(TWAction.PUTDOWN, null);
	
		} else if (curLocObj instanceof TWTile &&
				this.getEnvironment().canPickupTile((TWTile) curLocObj, this) &&
				this.carriedTiles.size() < 3) 
		{
//				planner.getGoals().add(new Int2D(this.x, this.y));
				return new TWThought(TWAction.PICKUP, null);
		} 
		
		
		// Goals Thought
		if (mode == Mode.FIND_FUEL_STATION) {
			// planner.voidGoals();
			// planner.voidPlan();
			if (this.planner.getGoals().isEmpty()) {
				// add goals
				addGoalsForFuelStation();
			}
			else {
				if (this.planner.getGoals().contains(new Int2D(this.x, this.y))) {
					// can get to the point
					int index = planner.getGoals().indexOf(new Int2D(this.x, this.y));
					if (index != -1) planner.getGoals().remove(0);
				}
				// can not get to but close
//				else {
//					int disToGoal = (int) this.getEnvironment().getDistance(this.x, this.y, planner.getGoals().get(0).getX(), planner.getGoals().get(0).getY());
//					if (disToGoal < 3) planner.getGoals().remove(0);
//				}				
			}
			
			for (int i = 0; i < planner.getGoals().size(); i++) {
				System.out.println("Goals " + i + ": " + planner.getGoals().get(i));			
			}	
		}
		else {
			planner.voidGoals();
			planner.voidPlan();
			if (mode == Mode.REFUEL) {
//				System.out.println(this.name + ": I am heading for the fuel station !" + "dis: " + this.getDistanceTo(memory.getFuelStation().x, memory.getFuelStation().y));
//				planner.getGoals().clear();
//				planner.voidPlan();
				planner.getGoals().add(memory.getFuelStation());
			}  else if ((mode == Mode.FILL)){
//				planner.getGoals().clear();
//				planner.voidPlan();
//				System.out.println(this.name + ": I am heading for a hole !!!!!!! " );
				planner.getGoals().add(new Int2D(holeTarget.getX(), holeTarget.getY()));
			} else if ((mode == Mode.COLLECT)){
//				planner.getGoals().clear();
//				planner.voidPlan();
//				System.out.println(this.name + ": I am heading for a tile !!!!!!! " );
				planner.getGoals().add(new Int2D(tileTarget.getX(), tileTarget.getY()));
			} else if (mode == Mode.WAIT){
				return new TWThought(TWAction.MOVE, TWDirection.Z);
			} else if (mode == Mode.EXPLORE) {
				return RandomMoveThought();
			}
//				Int2D randomPos = this.getEnvironment().generateRandomLocation();
//				Int2D randomPos = this.getEnvironment().generateFarRandomLocation(this.getX(), this.getY(), 2);
//				planner.getGoals().add(randomPos);
//				return new TWThought(TWAction.MOVE, RandomMoveThought());
//				return new TWThought(TWAction.MOVE, getRandomDirection());
				
		}
		
		for (int i = 0; i < planner.getGoals().size(); i++) {
//			System.out.println("Goals " + i + ": " + planner.getGoals().get(i));
		}	
		if (this.planner.getGoals().isEmpty()){
			return RandomMoveThought();
		}	
		
		this.planner.generatePlan();
		
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
		
		TWDirection dir = this.planner.execute();
		return new TWThought(TWAction.MOVE, dir);
		
//		return new TWThought(TWAction.MOVE, planner.execute());
		
//		planner.getGoals().add(this.getEnvironment().generateFarRandomLocation(this.getX(), this.getY(), 10));

//		} else if (mode == Mode.FILL) {
//			// 获取neighbour_hole, add to goal
//		} else if (mode == Mode.COLLECT) {
//			// 获取neighbour_tile, add to goal
//		} 

		
//		System.out.println("---------------------------------------" );
//		System.out.println( this.name +"Score: " + this.score);
//        return new TWThought(TWAction.MOVE, getRandomDirection());
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
//	this.planner.voidGoals();
//	this.planner.voidPlan();
	
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
		posY += Parameters.defaultSensorRange * 2+1;
		depth = depth + Parameters.defaultSensorRange * 2+1;
		if (depth > Parameters.xDimension/2) {
			break;
		}
		position = new Int2D (position.x, posY);
		planner.getGoals().add(getPositionAdd(base, position));
		
		// point 3
		posX = Parameters.defaultSensorRange;
		position = new Int2D (posX, position.y);
		planner.getGoals().add(getPositionAdd(base, position));
		depth = depth + (Parameters.defaultSensorRange * 2+1);
		
		// point 4
		posY += Parameters.defaultSensorRange * 2+1;
		position = new Int2D (position.x, posY);
	}
}


//	protected Object getClosest() {
//		TWTile tile = (TWTile) this.memory.closestInSensorRange.get(TWTile);
//		TWHole hole = (TWHole) this.memory.closestInSensorRange.get(TWHole);
//		return 
//	}

	private ArrayList<TWEntity> detectObjectNoCollision(PriorityQueue<TWEntity> neighborObjects, List<TWAgent> neighbouringAgents) {
		// TODO Auto-generated method stub
		ArrayList<TWEntity> tiles = new ArrayList<TWEntity>();
		while(!neighborObjects.isEmpty()) {
			TWEntity tile = neighborObjects.poll();
			for (int i = 0; i < neighbouringAgents.size(); i++) {
				// agent tile, agent inspector
				if (!(neighbouringAgents.get(i) instanceof MyInspectorAgent) 
						&& this.getDistanceTo(tile)/3 >= neighbouringAgents.get(i).getDistanceTo(tile)) {
					continue;
				} 
				tiles.add(tile);
			}
		}
				// if dis < agentDis && tile is not agent closest
//				if (this.getDistanceTo(tile) < neighbouringAgents.get(i).getDistanceTo(tile)) {
//					TWEntity closestHole = neighbouringAgents.get(i).memory.getClosestObjectInSensorRange(tile.getClass());
//					if (!tile.equals(closestHole)) {
//						return tile; 
//					}
//				}						
		if (tiles.size() > 0) return tiles;
		else return null;
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
				planner.getGoals().clear();
				break;
			case PUTDOWN:
				TWHole hole = (TWHole) memory.getMemoryGrid().get(this.x, this.y);
				putTileInHole(hole);
				planner.getGoals().clear();
				break;
			case REFUEL:
				refuel();
				planner.getGoals().clear();
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
		System.out.println("Step " + this.getEnvironment().schedule.getSteps());
		System.out.println(name + " score: " + this.score);
//		System.out.println("Assigned Zone: " + Integer.toString(agentZones[agentIdx]));
//		System.out.println("Mode: " + mode.name());
		System.out.println("Position: " + Integer.toString(this.x) + ", " + Integer.toString(this.y));
		System.out.println("Current Mode: " + this.mode);
		System.out.println("Size of goal: "+this.planner.getGoals().size());
		
		if (curGoal != null) {
			System.out.println("Goal: " + curGoal.x + ", " + curGoal.y);
		}
		else
			System.out.println("Goal: Nothing");
		System.out.println("Tiles: " + this.carriedTiles.size());
		System.out.println("Fuel Level: " + this.fuelLevel);
		System.out.println("Fuel Station: " + this.memory.getFuelStation());
		System.out.println("");

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
	
//	private TWDirection getRandomDirection(int X, int Y){
//	
//	    TWDirection randomDir = TWDirection.values()[this.getEnvironment().random.nextInt(5)];
//	
//	    if(this.getX()>=this.getEnvironment().getxDimension() ){
//	        randomDir = TWDirection.W;
//	    }else if(this.getX()<=1 ){
//	        randomDir = TWDirection.E;
//	    }else if(this.getY()<=1 ){
//	        randomDir = TWDirection.S;
//	    }else if(this.getY()>=this.getEnvironment().getxDimension() ){
//	        randomDir = TWDirection.N;
//	    }
//	
//	   return randomDir;
//	
//	}
	
	
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