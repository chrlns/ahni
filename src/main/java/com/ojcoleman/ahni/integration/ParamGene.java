package com.ojcoleman.ahni.integration;

import org.jgapcustomised.Gene;

/**
 * A generic parameter gene. A ParamGene may belong to a
 * {@link ParamCollection}. A ParamGene may have an index within the
 * ParamCollection.
 */
public class ParamGene extends Gene {

    private ParamCollection collection;
    private int index;

    /**
     * Create a ParamGene with the specified innovation ID (used by NEAT).
     * @param innovationId
     */
    public ParamGene(long innovationId) {
        super(innovationId);
    }

    /**
     * Create a ParamGene with the specified innovation ID (used by NEAT) that
     * belongs to the specified ParamCollection.
     * @param innovationId
     * @param collection
     * @param indexInCollection
     */
    public ParamGene(long innovationId, ParamCollection collection, int indexInCollection) {
        super(innovationId);
        this.collection = collection;
        this.index = indexInCollection;
    }

    public int getIndexWithinCollection() {
        return index;
    }

    public ParamCollection getCollection() {
        return collection;
    }

    /**
     * Get the minimum allowable value for this gene.
     *
     * @return 
     * @throws IllegalStateException if this gene does not belong to a
     * collection.
     */
    public double getMinValue() {
        if (collection == null) {
            throw new IllegalStateException("Cannot determine minimum allowable value for ParamGene when it does not belong to a ParamCollection.");
        }
        return collection.getMinOrMaxValueForGeneAtIndex(ParamCollection.MIN_VALUE, index);
    }

    /**
     * Get the maximum allowable value for this gene.
     *
     * @return 
     * @throws IllegalStateException if this gene does not belong to a
     * collection.
     */
    public double getMaxValue() {
        if (collection == null) {
            throw new IllegalStateException("Cannot determine maximum allowable value for ParamGene when it does not belong to a ParamCollection.");
        }
        return collection.getMinOrMaxValueForGeneAtIndex(ParamCollection.MAX_VALUE, index);
    }

    @Override
    public String toString() {
        return super.toString() + (collection != null ? " (" + index + "->" + collection + ")" : "");
    }
}
