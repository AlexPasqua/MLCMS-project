
package org.vadere.state.attributes.processor;

/**
 * attributes class regarding important fields for KNN, namely the measurement area and the KNN numbers to be detected
 */
public class KNNAttributes extends AttributesProcessor {

    // Variables
    private int measureAreaId = -1;
    private int kNN_num = -1;

    // Getter
    public int getMeasureAreaId() {
        return this.measureAreaId;
    }

    public int kNearestNeighbors() {
        return this.kNN_num;
    }
}

