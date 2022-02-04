package org.vadere.simulator.projects.dataprocessing.datakey;
import org.vadere.util.geometry.shapes.VPoint;
import java.util.List;

/**
 * Data structure containing all data about pedestrians: mean spacing between KNN and current pedestrian,
 * current pedestrian position, KNN positions
 */
public class KNNPedestrianData {
    private double meanSpacing;
    private VPoint pedestrianPosition;
    private List<VPoint> kNN;


    public KNNPedestrianData(double meanSpacing, VPoint pp, List<VPoint> kNN) {
        this.meanSpacing = meanSpacing;
        this.pedestrianPosition = pp;
        this.kNN = kNN;
    }

    public VPoint getPedestrianPosition() {
        return pedestrianPosition;
    }

    public String[] toStrings() {
        /**
         * fundamental for correct output to file
         */
        String[] ret = new String[kNN.size() + 2];
        ret[0] = meanSpacing + "";
        int i = 1;
        for (VPoint v : kNN) {
            ret[i] = v.x + " " + v.y;
            i++;
        }
        ret[ret.length-1] = pedestrianPosition.x + " " + pedestrianPosition.y;

        return ret;

    }

    @Override
    public String toString() {
        return "KNNPedestrianData{" +
                "meanSpacing=" + meanSpacing +
                ", pedestrianPosition=" + pedestrianPosition +
                ", kNN=" + kNN +
                '}';
    }
}
