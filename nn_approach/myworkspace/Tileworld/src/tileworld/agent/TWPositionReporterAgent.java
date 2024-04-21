package tileworld.agent;

import tileworld.environment.TWDirection;
import tileworld.environment.TWEnvironment;
import tileworld.environment.TWFuelStation;
import tileworld.exceptions.CellBlockedException;

/**
 * TWPositionReporterAgent
 *
 * This agent extends TWAgent and focuses on reporting its current position
 * at each step of the simulation while having the option to move randomly.
 */
public class TWPositionReporterAgent extends TWAgent {
    private String name;

    public TWPositionReporterAgent(String name, int xpos, int ypos, TWEnvironment env, double fuelLevel) {
        super(xpos, ypos, env, fuelLevel);
        this.name = name;}

    @Override
    protected TWThought think() {
        System.out.println("Agent Name: " + getName());
        System.out.println("Current Position: (" + getX() + ", " + getY() + ")");
        System.out.println("Simple Score: " + this.score);

        x = getX() + 1;
        y = getY() + 1;
        if (x >= getEnvironment().getxDimension()) {
            x = 0;
        }
        if (y >= getEnvironment().getyDimension()) {
            y = 0;
        }
        
        Object obj = this.getEnvironment().getObjectGrid().get(x, y);
        if (obj != null) {
            System.out.println("Object at position: " + obj);
        }


        
        return new TWThought(TWAction.MOVE, getRandomDirection());
    }

    @Override
    protected void act(TWThought thought) {

        //You can do:
        //move(thought.getDirection())
        //pickUpTile(Tile)
        //putTileInHole(Hole)
        //refuel()

        try {
            this.move(thought.getDirection());
        } catch (CellBlockedException ex) {

           // Cell is blocked, replan?
        }
    }

    private TWDirection getRandomDirection() {
        TWDirection randomDir = TWDirection.values()[getEnvironment().random.nextInt(TWDirection.values().length)];
        if (getX() >= getEnvironment().getxDimension()) {
            randomDir = TWDirection.W;
        } else if (getX() <= 1) {
            randomDir = TWDirection.E;
        } else if (getY() <= 1) {
            randomDir = TWDirection.S;
        } else if (getY() >= getEnvironment().getxDimension()) {
            randomDir = TWDirection.N;
        }
        return randomDir;
    }

    @Override
    public String getName() {
        return name;
    }

}
