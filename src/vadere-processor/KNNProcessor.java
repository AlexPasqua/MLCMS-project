package org.vadere.simulator.projects.dataprocessing.processor;

import org.vadere.annotation.factories.dataprocessors.DataProcessorClass;
import org.vadere.simulator.control.simulation.SimulationState;
import org.vadere.simulator.projects.dataprocessing.ProcessorManager;
import org.vadere.simulator.projects.dataprocessing.datakey.KNNPedestrianData;
import org.vadere.simulator.projects.dataprocessing.datakey.TimestepPedestrianIdKey;
import org.vadere.state.attributes.processor.KNNAttributes;
import org.vadere.state.attributes.processor.AttributesProcessor;
import org.vadere.state.scenario.MeasurementArea;
import org.vadere.state.scenario.Pedestrian;
import org.vadere.util.geometry.shapes.VPoint;

import java.util.*;

/**
 * the processor aim to give as output a file where each row is composite of the timeStep, pedestrian reference id,
 * mean spacing between reference and nearest neighbours, as well as the nearest neighbours themselves.
 */
@DataProcessorClass()
public class KNNProcessor extends DataProcessor<TimestepPedestrianIdKey, KNNPedestrianData> {

    public KNNProcessor() {
        super("meanSpacing", "kNearestNeighbors", "pedestrianPosition");
    }

    private MeasurementArea measurementArea;

    private int kNN;

    /** note that the k-nearest neighbors are calculated for pedestrians in a given measurement area**/
    @Override
    public void init(final ProcessorManager manager) {
        super.init(manager);
        KNNAttributes processorAttributes = (KNNAttributes) this.getAttributes();
        boolean rectangularAreaRequired = true;
        measurementArea = manager.getMeasurementArea(processorAttributes.getMeasureAreaId(), rectangularAreaRequired);
        kNN = processorAttributes.kNearestNeighbors();
    }

    @Override
    public AttributesProcessor getAttributes() {
        if (super.getAttributes() == null) {
            setAttributes(new KNNAttributes());
        }

        return super.getAttributes();
    }

    /**
     * function to update the knn of pedestrians in the measurement area
     * the doUpdate creates and writes all then needed rows in a given instant. One per each pedestrian having at least
     * KNN_num neighbours in the given measurement area
     */
    @Override
    protected void doUpdate(SimulationState state) {
        //create list of all pedestrians currently in the measurement area
        Collection<Pedestrian> pedestriansInMeasurementArea = new ArrayList<>();
        for (Pedestrian p : state.getTopography().getPedestrianDynamicElements().getElements()) {
            if (measurementArea.getShape().contains(p.getPosition())) {
                pedestriansInMeasurementArea.add(p);
            }
        }

        //loop over all pedestrians in area
        for (Pedestrian p : pedestriansInMeasurementArea) {

            PriorityQueue<DistanceCoordinates> pq = new PriorityQueue(kNN, new DistanceCoordinatesComparator());

            //loop over the neighborhood for the given pedestrian
            for (Pedestrian neighbor : state.getTopography().getPedestrianDynamicElements().getElements()) {
                if (p.equals(neighbor)) {
                    continue;
                }

                //create relative coordinates as asked by the paper, as well as structure for ordering positions
                double distance = getDistance(p.getPosition(), neighbor.getPosition());
                VPoint relativeCoords = new VPoint(neighbor.getPosition().x - p.getPosition().x, neighbor.getPosition().y - p.getPosition().y);
                DistanceCoordinates tmp = new DistanceCoordinates(distance, relativeCoords);

                //if structure is not full then add the neighbour
                if (pq.size() < kNN){
                    pq.add(tmp);
                    continue;
                }

                //if the neighbor is farther than most distance nearest neighbour then skip
                if (tmp.distance >= pq.peek().distance){
                    continue;
                }

                //discard the most distant nearest neighbour and substitute it
                pq.poll();
                pq.add(tmp);
            }
            //get mean spacing
            double meanSpacing = calculateMeanSpacingDistance(p, pq);

            //create the structure useful to output the results to data which is written to file
            ArrayList<VPoint> kneighbors = priorityQueueToArrayList(pq);
            KNNPedestrianData kNN = new KNNPedestrianData(meanSpacing, p.getPosition(), kneighbors);

            //putting the generated data for this pedestrian into the output file (key,value), if enough knn are there
            if(kneighbors.size() == this.kNN){
                System.out.println(kNN.getPedestrianPosition());
                putValue(new TimestepPedestrianIdKey(state.getStep(), p.getId()), kNN);
            }
        }


    }
    /** calculating mean spacing as sum of distances divided by number of neighbours**/
    private double calculateMeanSpacingDistance(Pedestrian p, PriorityQueue<DistanceCoordinates> pq) {
        double ret = 0;
        for (DistanceCoordinates dc : pq) {
            ret += dc.distance;
        }
        ret /= pq.size();
        return ret;
    }


    /** needed for the output **/
    public String[] toStrings(final TimestepPedestrianIdKey key) {
        return this.hasValue(key) ? this.getValue(key).toStrings() : new String[]{"N/A", "N/A"};
    }

    /** go from priority queue to array list ordered from nearest to most distant point **/
    private ArrayList<VPoint> priorityQueueToArrayList(PriorityQueue<DistanceCoordinates> pq) {
        ArrayList<VPoint> list = new ArrayList<>();
        for (int i = 0; i < this.kNN; i++) {
            DistanceCoordinates elem = pq.poll();
            if (elem!=null) {
                list.add(elem.coords);
            }
        }
        Collections.reverse(list);
        return list;

    }

    /** simple euclidian distance between two points **/
    private double getDistance(VPoint pos1, VPoint pos2) {
        return Math.sqrt(Math.pow((pos1.x - pos2.x), 2) + Math.pow((pos1.y - pos2.y), 2));
    }

    @Override
    public KNNPedestrianData getValue(TimestepPedestrianIdKey key) {
        KNNPedestrianData val = super.getValue(key);
        return val;
    }

    /** distance coordinates class to help create mean spacing **/
    private class DistanceCoordinates {
        public double distance;
        public VPoint coords;

        public DistanceCoordinates(double distance, VPoint coords) {
            this.distance = distance;
            this.coords = coords;
        }
    }

    /** used for the priority queue, needed for ordering and finding max distance in top 10 nearest**/
    private class DistanceCoordinatesComparator implements Comparator<DistanceCoordinates> {
        @Override
        public int compare(DistanceCoordinates a, DistanceCoordinates b) {
            return a.distance < b.distance ? 1 : a.distance == b.distance ? 0 : -1;
        }

    }
}

