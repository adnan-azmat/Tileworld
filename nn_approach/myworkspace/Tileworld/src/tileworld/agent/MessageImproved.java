package tileworld.agent;
import tileworld.environment.TWObject;
import tileworld.environment.TWObstacle;
import tileworld.environment.TWFuelStation;

import sim.util.Bag;
import sim.util.Int2D;
import sim.util.IntBag;

public class MessageImproved extends Message {
	
	private Bag sensedObjects;
	private Bag sensedAgents;
	private Double sensedTime;
	private Integer agentX;
	private Integer agentY;
	
	public MessageImproved(String from, String to, String message, Integer agentX, Integer agentY, Bag sensedObjects, Bag sensedAgents, double sensedTime){
		super(from, to, message);
		
		this.sensedObjects = sensedObjects;
		this.sensedAgents = sensedAgents;
		this.sensedTime = sensedTime;
		this.agentX = agentX;
		this.agentY = agentY;
	}
	
    public Bag getSensedObjects() {
        return this.sensedObjects;
    }
    
    public Bag getSensedAgents() {
        return this.sensedAgents;
    }
    
    public Double getSensedTime() {
        return this.sensedTime;
    }
    
    public Integer getAgentX() {
        return this.agentX;
    }
    
    public Integer getAgentY() {
        return this.agentY;
    }
}
