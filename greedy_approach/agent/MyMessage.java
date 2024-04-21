/**
 * 
 */
package tileworld.agent;

import sim.util.Bag;
import sim.util.Int2D;

/**
 * @author zengzheng
 *
 */
public class MyMessage extends Message {

	/**
	 * @param from
	 * @param to
	 * @param message
	 */
	Int2D fuelStation;
	Bag sensedObjects;
	Int2D posAgent;
	
//	Object objects [];
	public MyMessage(String from, String to, String message, Int2D fuelStation) {
		super(from, to, message);
		// TODO Auto-generated constructor stub
		this.fuelStation = fuelStation;
	}
	public MyMessage(String from, String to, String message, Bag sensedObjects, Int2D posAgent) {
		super(from, to, message);
		// TODO Auto-generated constructor stub
		this.sensedObjects = sensedObjects;
		this.posAgent = posAgent;
	}
	public Int2D getFuelStation() {
		return fuelStation;
	}
	
	public Bag getSensedObjects() {
		return sensedObjects;
	}
	
	public Int2D getPosAgent() {
		return posAgent;
	}

}