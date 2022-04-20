/*
 *   YAHNI Yet Another HyperNEAT Implementation
 *   Copyright (C) 2020-2022  Christian Lins <christian@lins.me>
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
package com.anji.neat;

import com.anji.integration.AnjiActivator;
import com.anji.integration.AnjiNetTranscriber;
import com.anji.integration.TranscriberException;
import com.anji.nn.AnjiNet;
import com.anji.nn.Connection;
import com.anji.nn.Neuron;
import com.anji.nn.NeuronConnection;
import com.anji.nn.activationfunction.DifferentiableFunction;
import com.ojcoleman.ahni.integration.ParamAllele;
import java.util.ArrayDeque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
import java.util.SortedMap;
import me.lins.yahni.neat.TrainingData;
import org.apache.commons.math3.util.Pair;
import org.jgapcustomised.Allele;
import org.jgapcustomised.ChromosomeMaterial;
import org.jgapcustomised.Configuration;
import org.jgapcustomised.InvalidConfigurationException;
import org.jgapcustomised.MutationOperator;

/**
 *
 * @author Christian Lins
 */
public class BackpropagationOperator extends MutationOperator {

    private boolean finalRun = false; // Run only at last
    private double learningRate = 0.01;
    
    public BackpropagationOperator() {
        super(1.0); // always
    }
    
    public void setFinalRun(boolean finalRun) {
        this.finalRun = finalRun;
    }
    
    public void setLearningRate(double a) {
        learningRate = a;
    }
    
    private Set<NeuronConnection> findOutConns(
            Map<Neuron,Set<NeuronConnection>> followUpNodes, Neuron neuron) 
    {
        if (followUpNodes.containsKey(neuron)) {
            return followUpNodes.get(neuron);
        } else {
            Set<NeuronConnection> conns = new HashSet<>();
            followUpNodes.put(neuron, conns);
            return conns;
        }
    }
    
    private void addOutConn(Map<Neuron,Set<NeuronConnection>> followUpNodes, 
            Neuron neuron, NeuronConnection conn)
    {
        if (followUpNodes.containsKey(neuron)) {
            followUpNodes.get(neuron).add(conn);
        } else {
            Set<NeuronConnection> conns = new HashSet<>();
            conns.add(conn);
            followUpNodes.put(neuron, conns);
        }       
    }
    
    @Override
    protected void mutate(Configuration config, ChromosomeMaterial chromeMat, 
            Set<Allele> allelesToAdd, Set<Allele> allelesToRemove, int currentGen, int maxGen) 
                throws InvalidConfigurationException 
    {
        if (finalRun && maxGen - currentGen > 1)
            return;
        
        Set<ParamAllele> paramAlleles = chromeMat.getAlleles(new ParamAllele(null));
        
        try {
            if (!(config.getBulkFitnessFunction() instanceof TrainingData))
                return;
            
            TrainingData data = (TrainingData)config.getBulkFitnessFunction();
            
            AnjiNetTranscriber transcriber = new AnjiNetTranscriber();
            AnjiNet net = transcriber.newAnjiNet(chromeMat.getAlleles(), "NULL");
            AnjiActivator activator = new AnjiActivator(net, 1);

            // Batch learning: we sum the weight changes for every data sample
            // for every connection and apply it afterwards
            Map<NeuronConnection,Double> ΔW = new HashMap<>();
            Map<Neuron,Set<NeuronConnection>> followUpNodes = new HashMap<>();
 
        /*    System.out.println("Vor Lernen:");
            for (int n = 0; n < 4; n++) {
                System.out.println("ist = " + 
                        Arrays.toString(activator.next(data.getInputData().get(n)))+ "\tsoll = " +
                        Arrays.toString(data.getOutputData().get(n)));
            }*/
            
            for (int n = 0; n < data.getInputData().size(); n++) {
                var inputData  = data.getInputData().get(n);
                var outputData = data.getOutputData().get(n);
                activator.next(inputData);
                
                // Queue of neurons that are about to be visited
                Queue<Pair<Neuron,Double>> backPropQueue = new ArrayDeque<>();
                
                // Set of neurons that have been visited and fully processed
                Set<Neuron> backPropVisited = new HashSet<>();
                
                // Start the backpropagation with the output neurons
                for (int m = 0; m < net.getOutputNeurons().size(); m++) {
                    Pair<Neuron,Double> pair = new Pair<>(
                        net.getOutputNeuron(m), outputData[m]);
                    backPropQueue.add(pair);
                }

                while (!backPropQueue.isEmpty()) {
                    var neuronPair = backPropQueue.remove();
                    var neuron = neuronPair.getFirst();
                    var target = neuronPair.getSecond();
                    var φ = neuron.getFunc();
                    backPropVisited.add(neuron);
                    
                    if (φ instanceof DifferentiableFunction) {
                        double dφ = ((DifferentiableFunction) φ)
                                .applyDiff(neuron.getNetInput());
                        double δ;
                        if (net.getOutputNeurons().contains(neuron)) {
                            δ = dφ * (neuron.getValue() - target);
                            neuron.setErrorSignal(δ);
                        } else {
                            double sum = 0;

                            // Get all connections of the next layer that are
                            // connected to this neuron
                            var iconns = followUpNodes.get(neuron);
                            for (Connection iconn : iconns) {
                                NeuronConnection nconn = (NeuronConnection) iconn;
                                double w = nconn.getWeight();
                                sum += nconn.getOutgoingNode().getErrorSignal() * w;
                            }

                            δ = dφ * sum;
                            neuron.setErrorSignal(δ);
                        }
                        
                        for (var conn : neuron.getIncomingConns()) {
                            NeuronConnection nconn = (NeuronConnection)conn;
                            // TODO und weight? und bias?
                            double Δw = -learningRate * δ 
                                    * nconn.getIncomingNode().getValue();
                            if (!ΔW.containsKey(nconn)) {
                                ΔW.put(nconn, 0.0); 
                            }
                            ΔW.put(nconn, ΔW.get(nconn) + Δw);
                        }
                    }

                    // Add the next neurons to the backpropagation queue
                    for (Connection conn : neuron.getIncomingConns()) {
                        NeuronConnection nconn = (NeuronConnection) conn;
                        Neuron neu = nconn.getIncomingNode();
                        
                        addOutConn(followUpNodes, neu, nconn);
                        
                        if (!backPropVisited.contains(neu) && !net.getInputNeurons().contains(neu)) {
                            Pair<Neuron,Double> pair = new Pair<>(neu, 0.0);
                            backPropQueue.add(pair);
                        }
                    }
                }

            }
            
            SortedMap<Long,ConnectionAllele> alleles = NeatChromosomeUtility
                    .getConnectionMap(chromeMat.getAlleles());

            // Batch learning completed, apply ΔW changes to connections
            // TODO We could avoid ΔW and apply the changes directly to the
            // genotype as this only re-read in the next generation
            if (!ΔW.isEmpty()) {
                net.getAllNeurons().forEach((neuron) -> {
                    for (var conn : neuron.getIncomingConns()) {
                        if (conn instanceof NeuronConnection) {
                            NeuronConnection nconn = (NeuronConnection)conn;
                            ConnectionAllele call = alleles.get(nconn.getId());
                            if (ΔW.containsKey(nconn)) {
                                double Δw = ΔW.get(nconn);
                                call.setWeight(call.getWeight() + Δw);
                                nconn.applyΔW(Δw); // For debugging
                            } 
                        }
                    }
                });
            }
                
        } catch(TranscriberException ex) {
            throw new InvalidConfigurationException(ex.toString());
        }
    }
    
}
