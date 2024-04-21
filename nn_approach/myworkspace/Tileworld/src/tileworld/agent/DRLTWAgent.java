/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package tileworld.agent;

import tileworld.environment.TWTile;
import tileworld.environment.TWHole;
import tileworld.environment.TWObject;
import tileworld.environment.TWFuelStation;

import java.util.AbstractMap.SimpleEntry;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

import sim.util.Bag;
import sim.util.Double2D;
import sim.util.Int2D;
import tileworld.environment.TWDirection;
import tileworld.environment.TWEntity;
import tileworld.environment.TWEnvironment;
import tileworld.exceptions.CellBlockedException;
import tileworld.Parameters;
import tileworld.agent.MessageImproved;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;

import java.util.Iterator;
import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;

import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.TranslateException;
import ai.djl.training.listener.TrainingListener;
// import ai.djl.basicdataset.RandomAccessDataset;

import ai.djl.util.PairList;
import ai.djl.training.initializer.Initializer;
import ai.djl.translate.Batchifier;

import ai.djl.MalformedModelException;


/**
 * TWContextBuilder
 *
 * @author lennardkorte
 *         Created: Mar 31, 2024
 *
 *         Copyright lennardkorte Expression year is undefined on line 16,
 *         column 24 in Templates/Classes/Class.java.
 *
 *
 *         Description:
 *
 */
public class DRLTWAgent extends TWAgent {
    double reward_total;
    private String name;
    protected TWAgentWorkingMemoryImproved memory;
    private TWDirection lastAction = null; // Variable to hold the last action performed
    LinkedList<Double> lastDirections = new LinkedList<>();
    private Model model = null;
    private String parameterFilePath = null;
    private NDManager manager = NDManager.newBaseManager();

    public DRLTWAgent(String name, int xpos, int ypos, TWEnvironment env, double fuelLevel, String parameterFilePath) {
        super(xpos, ypos, env, fuelLevel);
        this.reward_total = 0;
        this.name = name;
        this.memory = new TWAgentWorkingMemoryImproved(this, env.schedule, env.getxDimension(), env.getyDimension());
        this.lastDirections = new LinkedList<>();
        while (lastDirections.size() < 3) {
            lastDirections.add((double) getEnvironment().random.nextInt(4) + 1);
        }
        
        this.model = instantiateModel();
        this.parameterFilePath = "model_parameters";
        // loadModelParameters(model, parameterFilePath);
        

    }

    public String getName() {
        return name;
    }

    @Override
    public TWAgentWorkingMemoryImproved getMemory() {
        return memory;
    }

    @Override
    public void communicate() {
        ArrayList<Message> messages = getEnvironment().getMessages();
        messages.removeIf(message -> message.getFrom().equals(getName()));

        for (int i = 0; i < messages.size(); i++) {
            MessageImproved msg = (MessageImproved) messages.get(i); // unsafe cast
            memory.updateMemoryAt(msg.getAgentX(), msg.getAgentY(), msg.getSensedObjects(), msg.getSensedAgents(),
                    msg.getSensedTime());
        }
        MessageImproved new_message = new MessageImproved(getName(), "", "", getX(), getY(), memory.getSensedObjects(),
                memory.getSensedAgents(), memory.getSensedTime());
        getEnvironment().receiveMessage(new_message);

        for (Message message : messages) {
            // System.out.println("message from" + message.getFrom() + " " +
            // messages.size());
        }
    }

    @Override
    protected TWThought think() {
        TWThought thought = null; // Variable to hold the decided action
        lastAction = getRandomDirection();

        // Retrieve the nearby percepts from memory
        Map<String, Map<Int2D, Double2D>> nearbyPercepts = getMemory().getNearbyPercepts();
        System.out.println(nearbyPercepts);

        // Add the current direction to the list
        lastDirections.add((double) getPreviousMovement());

        // Save the last 3 directions
        if (lastDirections.size() > 3) {
            lastDirections.removeFirst();
        }

        System.out.println("Last Directions: " + lastDirections);

        // get the fuel station location (Relative)
        Map<Int2D, Double2D> relativePosition = new HashMap<>();
        TWFuelStation fuelStationLoc = getNearbyFuelStation(getX(), getY());
        if (fuelStationLoc != null) {
            int relativeX = fuelStationLoc.getX() - getX();
            int relativeY = fuelStationLoc.getY() - getY();

            double manhattanMagnitude = Math.abs(relativeX) + Math.abs(relativeY);
            double radialDistance = Math.atan2(relativeY, relativeX);
            relativePosition.put(new Int2D(relativeX, relativeY), new Double2D(manhattanMagnitude, radialDistance));
        } else {
            relativePosition.put(new Int2D(0, 0), new Double2D(10000, 0));
        }

        // get number of tiles carried
        double numTilesCarried = carriedTiles.size();
        System.out.println("Number of Tiles Carried: " + numTilesCarried);

        // First, we define some key checks and values
        boolean atFuelStation = false;
        TWFuelStation fuelStation = getNearbyFuelStation(getX(), getY());
        if (fuelStation != null) {
            atFuelStation = (getX() == fuelStation.getX() && getY() == fuelStation.getY());
        }
        boolean atTile = (numTilesCarried > 0.0 && memory.getNearbyTile(getX(), getY(), Double.MAX_VALUE) != null
                && getX() == memory.getNearbyTile(getX(), getY(), Double.MAX_VALUE).getX()
                && getY() == memory.getNearbyTile(getX(), getY(), Double.MAX_VALUE).getY());

        boolean atHole = (numTilesCarried > 0.0 && memory.getNearbyHole(getX(), getY(), Double.MAX_VALUE) != null
                && getX() == memory.getNearbyHole(getX(), getY(), Double.MAX_VALUE).getX()
                && getY() == memory.getNearbyHole(getX(), getY(), Double.MAX_VALUE).getY());

        double currentFuel = getFuelLevel();
        boolean isCarryingTiles = hasStorage();
        int minFuelNeeded = (int) (Parameters.defaultFuelLevel * 0.9); // minimum fuel threshold for refueling (90% of max fuel capacity)

        // whenever on tile: perform respective action.
        if (atFuelStation && currentFuel < minFuelNeeded) {
            thought = new TWThought(TWAction.REFUEL, TWDirection.Z); // Z direction represents no movement
        } else if (atHole && isCarryingTiles) {
            thought = new TWThought(TWAction.PUTDOWN, TWDirection.Z); // Z direction represents no movement to fill the
                                                                      // hole
        } else if (atTile && !isCarryingTiles) {
            thought = new TWThought(TWAction.PICKUP, TWDirection.Z); // Z direction represents no movement to pick up
                                                                     // the tile
        }
        
        // NDArray input = manager.randomUniform(0f, 1f, new Shape(1, 22)); // Example input
        NDArray input = playerInput(manager, nearbyPercepts, relativePosition, lastDirections, numTilesCarried, currentFuel, (double) carriedTiles.size());
        
        TWDirection direction = predict(model, input);
        System.out.println("Prediction result: " + direction);

        //trainModel(model, parameterFilePath);
        
        if (thought == null) {
            // Default action if no other action is decided
            thought = new TWThought(TWAction.MOVE, direction);
        }

        double step_reward = reward(thought);
        // System.out.println(getName() + " Reward: " + step_reward + " " +
        // thought.getAction());
        reward_total += step_reward;

        return thought;
    }
    
    public NDArray playerInput(NDManager manager, Map<String, Map<Int2D, Double2D>> nearbyPercepts, Map<Int2D, Double2D> relativePosition,
                              LinkedList<Double> lastDirections2, double numTilesCarried, double currentFuel, double size) {

        List<Double> values = new ArrayList<>();

        // Add individual double values
        values.add(numTilesCarried);
        values.add(currentFuel);
        values.add(size);
        values.addAll(lastDirections2);

        // Extract Double2D values from relativePosition
        for (Double2D value : relativePosition.values()) {
            values.add(value.x);
            values.add(value.y);
        }

        // Extract Double2D values from nearbyPercepts
        for (Map<Int2D, Double2D> map : nearbyPercepts.values()) {
            for (Double2D value : map.values()) {
                values.add(value.x);
                values.add(value.y);
            }
        }

        // Create an INDArray with all the collected values
        double[] primitiveValues = values.stream().mapToDouble(Double::doubleValue).toArray();
        // Use NDManager to create an NDArray
        NDArray inputArray = manager.create(primitiveValues);


        return inputArray;
    }

    private Model instantiateModel() {
        Model model = Model.newInstance("simple-softmax-model");
        Block block = new SequentialBlock()
                .add(Linear.builder().setUnits(50).build())
                .add(Activation.reluBlock())
                .add(Linear.builder().setUnits(20).build());
            
        // Sets the block to the model.
        model.setBlock(block);

        // Initialize the model's parameters using the centrally managed NDManager.
        model.getBlock().initialize(manager, DataType.FLOAT32, new Shape(1, 22));

        return model;
    }

    private TWDirection predict(Model model, NDArray input) {
        try (Predictor<NDList, NDList> predictor = model.newPredictor(new NoopTranslator())) {
            NDList output = predictor.predict(new NDList(input));
            NDArray softmaxResult = output.singletonOrThrow().softmax(1); // Applying softmax here
            long maxIndex = softmaxResult.argMax(1).getLong();
            TWDirection direction = TWDirection.values()[(int)maxIndex];
            System.out.println("Prediction output: " + softmaxResult);
            return direction;
        } catch (Exception e) {
            System.err.println("Prediction failed: " + e.getMessage());
            e.printStackTrace();
            return TWDirection.Z; // Default if prediction fails
        }
    }
    
    public static void trainModelMultipleEpochs(Model model, RandomAccessDataset trainingData, int epochs) {
        try (Trainer trainer = model.newTrainer(createTrainingConfig())) {
            trainer.initialize(new Shape(1, 2));
            Metrics metrics = new Metrics();
            trainer.setMetrics(metrics);

            for (int epoch = 0; epoch < epochs; epoch++) {
                System.out.println("Epoch " + (epoch + 1) + "/" + epochs);
                trainSingleEpoch(trainer, trainingData);
            }
        }
    }

    private static void trainSingleEpoch(Trainer trainer, RandomAccessDataset trainingData) {
        for (Batch batch : trainer.iterateDataset(trainingData)) {
            try {
                EasyTrain.trainBatch(trainer, batch);
                trainer.step();
            } catch (TranslateException e) {
                System.err.println("Error processing batch: " + e.getMessage());
                // Handle the exception, e.g., log it, skip this batch, etc.
            } finally {
                batch.close(); // Ensure batch is always closed
            }
        }
    }

    private static DefaultTrainingConfig createTrainingConfig() {
        return new DefaultTrainingConfig(Loss.l2Loss())
                .addTrainingListeners(TrainingListener.Defaults.logging());
    }
    
    private void trainModel(Model model, String parameterFilePath) {
    	RandomAccessDataset trainingData = createDummyDataset(manager);
        trainModelMultipleEpochs(model, trainingData, 5);  // Train for 5 epochs
        saveModelParameters(model, parameterFilePath);
    }
    
    private RandomAccessDataset createDummyDataset(NDManager manager) {
        int batchSize = 32; // Define the batch size for training
        int numSamples = 320; // Total number of samples in the dataset
        int featureSize = 22; // Number of features in each input vector, based on model initialization
    
        // Prepare arrays to hold the data and labels
        float[][] featureData = new float[numSamples][featureSize];
        float[] labelData = new float[numSamples];
    
        Random rand = new Random();
    
        // Generate random data and labels
        for (int i = 0; i < numSamples; i++) {
            for (int j = 0; j < featureSize; j++) {
                featureData[i][j] = rand.nextFloat(); // Populate features with random values
            }
            labelData[i] = rand.nextInt(5); // Assuming 5 possible actions (0 to 4)
        }
    
        // Create NDArrays from the data
        NDArray features = manager.create(featureData);
        NDArray labels = manager.create(labelData);
    
        // Create the dataset using ArrayDataset which allows setting of features and labels
        ArrayDataset dataset = new ArrayDataset.Builder()
                .setData(features) // Set the feature NDArrays
                .optLabels(labels) // Set the label NDArrays
                .setSampling(batchSize, true) // Set batch size and enable shuffle
                .build();
    
        return dataset;
    }
    
    
    // Method to save model parameters to a file
    public void saveModelParameters(Model model, String filePath) {
        try {
            model.save(Paths.get(filePath), "simple-softmax-model");
            System.out.println("Model parameters saved to: " + filePath);
        } catch (IOException e) {
            System.err.println("Failed to save model parameters: " + e.getMessage());
        }
    }

    // Method to load model parameters from a file
    private void loadModelParameters(Model model, String filePath) {
        Path path = Paths.get(filePath);
        if (Files.exists(path)) {
            try {
                model.load(path, "simple-softmax-model");
                System.out.println("Model parameters loaded from: " + filePath);
            } catch (IOException | MalformedModelException e) {  // Catch both exceptions here
                System.err.println("Failed to load model parameters: " + e.getMessage());
            }
        } else {
            System.out.println("No model parameters found at the specified path: " + filePath);
        }
    }

    private int getPreviousMovement() {

        if (lastAction == TWDirection.N) {
            return 1;
        } else if (lastAction == TWDirection.E) {
            return 2;
        } else if (lastAction == TWDirection.S) {
            return 3;
        } else if (lastAction == TWDirection.W) {
            return 4;
        } else {
            System.out.println("No previous movement");
            System.out.println(lastAction);
            // System.exit(1);
            return 0;/*  */
        }
    }

    public double reward(TWThought thought) {
        int reward_s = -1000;
        int reward_m = -1;
        int reward_l = 100;
        TWAction thought_action = thought.getAction();
        if (thought_action != TWAction.REFUEL && getFuelLevel() == 1) {
            return reward_s;
        }
        switch (thought_action) {
            case MOVE:
                // check if in bounds and if no obstacles and check for fuelLevel <= 0
                TWDirection direction = thought.getDirection();
                boolean cellBlocked = memory.isCellBlocked(getX() + direction.dx, getY() + direction.dy);
                // boolean inBounds = getEnvironment().isInBounds(getX() + direction.dx, getY()
                // + direction.dy);
                if (!cellBlocked && getFuelLevel() > 0) {
                    return reward_m;
                }
                return reward_s;

            case PICKUP:
                // is there a tile and is there still storage?
                TWTile tile = getTileAtCurrentLocation();
                if (tile != null && hasStorage()) {
                    return reward_l;
                }
                return reward_s;
            case PUTDOWN:
                // is there a tile in storage and is there a hole?
                TWHole hole = getHoleAtCurrentLocation();
                if (hole != null && hasTile()) {
                    return reward_l;
                }
                return reward_s;
            case REFUEL:
                // check if no fuel station and how full fuel tank is
                if (ifOnFuelStation(getX(), getY())) {
                    return -getFuelLevel();
                }
                return reward_s;
        }
        return 0;
    }

    @Override
    protected void act(TWThought thought) {
        try {
            switch (thought.getAction()) {
                case MOVE:
                    move(thought.getDirection());
                    lastAction = thought.getDirection();
                    break;
                case PICKUP:
                    pickUpTile(getTileAtCurrentLocation());
                    break;
                case PUTDOWN:
                    putTileInHole(getHoleAtCurrentLocation());
                    break;
                case REFUEL:
                    refuel();
                    break;
            }
            ;
        } catch (CellBlockedException e) {
            // Handle blocked cell, perhaps choose a different action or direction
        }
    }

    private TWTile getTileAtCurrentLocation() {
        Object obj = memory.getMemoryGrid().get(getX(), getY());
        if (obj instanceof TWTile) {
            return (TWTile) obj;
        }
        return null;
    }

    private TWHole getHoleAtCurrentLocation() {
        Object obj = memory.getMemoryGrid().get(getX(), getY());
        if (obj instanceof TWHole) {
            return (TWHole) obj;
        }
        return null;
    }

    public boolean hasStorage() {
        return carriedTiles.size() < 3;
    }

    /*
     * private int forwardPass(){
     * 
     * //TODO
     * 
     * //layers:
     * getEnvironment().getxDimension()
     * getEnvironment().getyDimension()
     * 
     * // layer with 1 to indicate a field is in the environment
     * // cut grid around current agent position by 129 total (64 each side):
     * //int memsize = memory.getMemorySize();
     * 
     * 
     * // layer with obstacles
     * 
     * // layer with tiles
     * 
     * // layer with holes
     * 
     * //ObjectGrid2D memorygrid = memory.getMemoryGrid();
     * 
     * // full layer indicating fuel level (full layer)
     * //double fuelLevel = getFuelLevel();
     * 
     * // layer indicating position of other agents
     * 
     * // layer indicating fuel level of other agents
     * 
     * // layer indicating storage of other agents
     * 
     * return 0;
     * 
     * }
     */

    // Only for test thinking
    // ##################################################################################
    // Only for test thinking
    // ##################################################################################
    // Only for test thinking
    // ##################################################################################

    private TWDirection getRandomDirection() {

        TWDirection randomDir = TWDirection.values()[getEnvironment().random.nextInt(4)];

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

    private TWDirection calculateDirectionTo(int targetX, int targetY) {
        int dx = targetX - getX();
        int dy = targetY - getY();

        if (Math.abs(dx) > Math.abs(dy)) {
            return dx > 0 ? TWDirection.E : TWDirection.W;
        } else {
            return dy > 0 ? TWDirection.S : TWDirection.N;
        }
    }

    public TWFuelStation getNearbyFuelStation(int x, int y) {
        TWFuelStation nearestFuelStation = null;
        double nearestDistance = Double.MAX_VALUE; // Start with the max value

        for (TWAgentPercept fuelStationPercept : memory.getFuelStations()) {
            TWFuelStation fuelStation = (TWFuelStation) fuelStationPercept.getO();
            double distance = calculateDistance(x, y, fuelStation.getX(), fuelStation.getY());
            if (distance < nearestDistance) {
                nearestDistance = distance;
                nearestFuelStation = fuelStation;
            }
        }

        return nearestFuelStation;
    }

    private int calculateDistance(int x1, int y1, int x2, int y2) {
        // Chebyshev distance
        return Math.max(Math.abs(x2 - x1), Math.abs(y2 - y1));
    }

    public boolean ifOnFuelStation(int x, int y) {
        for (TWAgentPercept fuelStationPercept : memory.getFuelStations()) {
            TWFuelStation fuelStation = (TWFuelStation) fuelStationPercept.getO();
            double distance = calculateDistance(x, y, fuelStation.getX(), fuelStation.getY());
            if (distance == 0) {
                return true;
            }
        }
        return false;
    }
}
