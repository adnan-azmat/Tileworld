package tileworld.agent;

import java.util.HashMap;
import java.util.Map;

import sim.util.Bag;
import sim.util.Int2D;

public class Message {
	/*
	 * Usage Tutorial
	 * 
	 * Method 1: Create an empty Message and send it
	 * 		Message msgFuel = new Message("", "");
	 * 		this.getEnvironment().receiveMessage(msgFuel);
	 * 
	 * Method 2: Create a Message containing one thing (eg. Int2D fuelStationPosition) and send it
	 * 		Message msgFuel = new Message("", "");
	 * 		Int2D fuel = new Int2D(this.memory.getFuelStation().getX(), this.memory.getFuelStation().getY());
	 * 		msgFuel.addFuelStationPosition(fuel);
	 * 		this.getEnvironment().receiveMessage(msgFuel);
	 * 
	 * Method 3: Create a Message containing many things
	 * 		Message msg = new Message("", "");
	 * 		Int2D fuel;
	 * 		Bag sensedObjects;
	 * 		msg.addFuelStationPosition(fuel);
	 * 		msg.addSensedObjects(sensedObjects);
	 * 		this.getEnvironment().receiveMessage(msgFuel);
	 * 
	 * Method 4: Receiving a Message
	 * 		for (Message m : messages) {
	 * 			if (m == null || m.message == null) continue;
	 * 			Int2D fuelStationPos = m.getFuelStationPosition();
	 * 			if (fuelStationPos != null) {
	 * 				this.memory.setFuelStation(fuelStationPos);
	 * 				break;
	 * 			}
	 * 		}
	 * 
	 */
	private String from; // the sender
	private String to; // the recepient
	/*
	 * String "fuelStationPosition": Object (Int2D) Position of FuelStation
	 * String "sensedObjects": Object (Bag) the objects that sensed by sensor (May contain null)
	 * String "agentPosition": the sending agent's position (may be used for the sensor's center to update memory)
	 */
	Map<String, Object> message;
	
	public Message(String from, String to) {
		this.from = from;
		this.to = to;
		this.message = new HashMap<String, Object>();
	}
	
	public Message(String from, String to, Map<String, Object> message){
		this.from = from;
		this.to = to;
		this.message = message;
	}

	public String getFrom() {
		return from;
	}

	public String getTo() {
		return to;
	}

	public Map<String, Object> getMessage() {
		return this.message;
	}
	
	public void addMessage(String Key, Object Value) {
		this.message.put(Key, Value);
	}
	
	public void addFuelStationPosition(Int2D fuelStationPosition) {
		this.addMessage("fuelStationPosition", fuelStationPosition);
	}
	
	public Int2D getFuelStationPosition() {
		if (this.message.containsKey("fuelStationPosition")) {
			return (Int2D) this.message.get("fuelStationPosition");
		}
		else return null;
	}
	
	public void addSensedObjects(Bag sensedObjects, Int2D agentPosition) {
		this.addMessage("sensedObjects", sensedObjects);
		this.addMessage("agentPosition", agentPosition);
	}
	
	public Bag getSensedObjects() {
		if (this.message.containsKey("sensedObjects")) {
			return (Bag) this.message.get("sensedObjects");
		}
		else return null;
	}
	
	public Int2D getAgentPosition() {
		if (this.message.containsKey("agentPosition")) {
			return (Int2D) this.message.get("agentPosition");
		}
		else return null;
	}
}




////package tileworld.agent;
////
////public class Message {
////	private String from; // the sender
////	private String to; // the recepient
////	private String message; // the message
////	
////	public Message(String from, String to, String message){
////		this.from = from;
////		this.to = to;
////		this.message = message;
////	}
////
////	public String getFrom() {
////		return from;
////	}
////
////	public String getTo() {
////		return to;
////	}
////
////	public String getMessage() {
////		return message;
////	}
////
////}
//
//
//
//package tileworld.agent;
//
//import java.util.HashMap;
//import java.util.Map;
//
//import sim.util.Bag;
//import sim.util.Int2D;
//
//public class Message {
//	/*
//	 * Usage Tutorial
//	 * 
//	 * Method 1: Create an empty Message and send it
//	 * 		Message msgFuel = new Message("", "");
//	 * 		this.getEnvironment().receiveMessage(msgFuel);
//	 * 
//	 * Method 2: Create a Message containing one thing (eg. Int2D fuelStationPosition) and send it
//	 * 		Message msgFuel = new Message("", "");
//	 * 		Int2D fuel = new Int2D(this.memory.getFuelStation().getX(), this.memory.getFuelStation().getY());
//	 * 		msgFuel.addFuelStationPosition(fuel);
//	 * 		this.getEnvironment().receiveMessage(msgFuel);
//	 * 
//	 * Method 3: Create a Message containing many things
//	 * 		Message msg = new Message("", "");
//	 * 		Int2D fuel;
//	 * 		Bag sensedObjects;
//	 * 		msg.addFuelStationPosition(fuel);
//	 * 		msg.addSensedObjects(sensedObjects);
//	 * 		this.getEnvironment().receiveMessage(msgFuel);
//	 * 
//	 * Method 4: Receiving a Message
//	 * 		for (Message m : messages) {
//	 * 			if (m == null || m.message == null) continue;
//	 * 			Int2D fuelStationPos = m.getFuelStationPosition();
//	 * 			if (fuelStationPos != null) {
//	 * 				this.memory.setFuelStation(fuelStationPos);
//	 * 				break;
//	 * 			}
//	 * 		}
//	 * 
//	 */
//	private String from; // the sender
//	private String to; // the recepient
//	/*
//	 * String "fuelStationPosition": Object (Int2D) Position of FuelStation
//	 * String "sensedObjects": Object (Bag) the objects that sensed by sensor (May contain null)
//	 * String "agentPosition": the sending agent's position (may be used for the sensor's center to update memory)
//	 */
//	Map<String, Object> message;
//	
//	public Message(String from, String to) {
//		this.from = from;
//		this.to = to;
//		this.message = new HashMap<String, Object>();
//	}
//	
//	public Message(String from, String to, Map<String, Object> message){
//		this.from = from;
//		this.to = to;
//		this.message = message;
//	}
//
//	public String getFrom() {
//		return from;
//	}
//
//	public String getTo() {
//		return to;
//	}
//
//	public Map<String, Object> getMessage() {
//		return this.message;
//	}
//	
//	public void addMessage(String Key, Object Value) {
//		this.message.put(Key, Value);
//	}
//	
//	public void addFuelStationPosition(Int2D fuelStationPosition) {
//		this.addMessage("fuelStationPosition", fuelStationPosition);
//	}
//	
//	public Int2D getFuelStationPosition() {
//		if (this.message.containsKey("fuelStationPosition")) {
//			return (Int2D) this.message.get("fuelStationPosition");
//		}
//		else return null;
//	}
//	
//	public void addSensedObjects(Bag sensedObjects, Int2D agentPosition) {
//		this.addMessage("sensedObjects", sensedObjects);
//		this.addMessage("agentPosition", agentPosition);
//	}
//	
//	public Bag getSensedObjects() {
//		if (this.message.containsKey("sensedObjects")) {
//			return (Bag) this.message.get("sensedObjects");
//		}
//		else return null;
//	}
//	
//	public Int2D getAgentPosition() {
//		if (this.message.containsKey("agentPosition")) {
//			return (Int2D) this.message.get("agentPosition");
//		}
//		else return null;
//	}
//}