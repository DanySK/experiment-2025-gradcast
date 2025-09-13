package it.unibo.gradcast

import it.unibo.alchemist.collektive.device.CollektiveDevice
import it.unibo.alchemist.model.Position
import it.unibo.collektive.aggregate.Field
import it.unibo.collektive.aggregate.api.Aggregate
import it.unibo.collektive.stdlib.spreading.distanceTo
import it.unibo.collektive.stdlib.spreading.gossipMin

/**
 * Main experiment function for GradCast.
 */
fun <P : Position<P>> Aggregate<Int>.experiment(device: CollektiveDevice<P>): Double {
//    val isSource: Boolean = device.randomGenerator.nextInt() % 50 == 0
    val metric = with(device) { distances() }
    return distanceTo(localId == 0 || localId == 1000 || localId == 100, metric)
    // return (bullsEye(metric) * 1000).toInt() / 1000.0
}

/**
 * Bull's eye pattern implementation (simplified for v27 compatibility).
 */
fun Aggregate<Int>.bullsEye(metric: Field<Int, Double>): Double {
    // Simplified version for v27 compatibility
    // Creates a gradient from a randomly chosen node (using gossipMin),
    // measuring distances based on the provided metric.
    val distToRandom = distanceTo(gossipMin(localId) == localId, metric)

    // TODO: Implement the full bullseye algorithm once v27 API is better understood
    // For now, just return the distance to the random node
    return distToRandom
}
