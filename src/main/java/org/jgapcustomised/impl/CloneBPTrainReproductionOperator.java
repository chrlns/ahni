/*
 *   YAHNI Yet Another HyperNEAT Implementation
 *   Copyright (C) 2022  Christian Lins <christian@lins.me>
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package org.jgapcustomised.impl;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.jgapcustomised.Chromosome;
import org.jgapcustomised.ChromosomeFitnessComparator;
import org.jgapcustomised.ChromosomeMaterial;
import org.jgapcustomised.Configuration;
import org.jgapcustomised.ReproductionOperator;

/**
 * Produces offspring be creating clones of parents and train them with
 * the backpropagation mutation operator.
 * 
 * TODO Is the offspring compared with parents afterwards?
 *
 * @author Christian Lins
 */
public class CloneBPTrainReproductionOperator extends ReproductionOperator {

    /**
     * Adds new children of <code>parents</code> to <code>offspring</code>.
     *
     * @param config
     * @param parents <code>List</code> contains Chromosome objects
     * @param numOffspring
     * @param offspring <code>List</code> contains ChromosomeMaterial objects
     * @see org.jgapcustomised.ReproductionOperator#reproduce(Configuration,
     * List, int, List)
     */
    @Override
    protected void reproduce(final Configuration config, final List<Chromosome> parents, int numOffspring, List<ChromosomeMaterial> offspring) {
        reproduce(parents, numOffspring, offspring);
    }

    /**
     * Adds new children of <code>parents</code> to <code>offspring</code>.
     *
     * @param parents <code>List</code> contains <code>Chromosome</code> objects
     * @param numOffspring
     * @param offspring <code>List</code> contains
     * <code>ChromosomeMaterial</code> objects
     */
    @SuppressWarnings("unchecked")
    public static void reproduce(final List<Chromosome> parents, int numOffspring, List<ChromosomeMaterial> offspring) {
        // Sort fittest first to ensure we include these (and more than once if numOffspring is greater than number of parents).
        List<Chromosome> parentsSorted = new ArrayList<>(parents);
        Collections.sort(parentsSorted, new ChromosomeFitnessComparator(false, false));
        for (int i = 0; i < numOffspring; i++) {
            Chromosome parent = parentsSorted.get(i % parentsSorted.size());
            offspring.add(parent.cloneMaterial());
        }
    }

}
