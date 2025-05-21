
package it.unibo.gradcast

import it.unibo.alchemist.collektive.device.CollektiveDevice
import it.unibo.alchemist.model.Position
import it.unibo.collektive.aggregate.Field
import it.unibo.collektive.aggregate.api.Aggregate
import it.unibo.collektive.stdlib.spreading.distanceTo
import it.unibo.collektive.stdlib.spreading.gossipMax
import it.unibo.collektive.stdlib.spreading.gossipMin
import it.unibo.collektive.stdlib.spreading.gradientCast
import kotlin.math.abs
import kotlin.math.hypot

fun <P: Position<P>> Aggregate<Int>.experiment(device: CollektiveDevice<P>): Double {
//    val isSource: Boolean = device.randomGenerator.nextInt() % 50 == 0
    val metric = with(device) { distances() }
//    return distanceTo(localId == 0, isRiemannianManifold = false, metric = metric)
    return (bullsEye(metric) * 1000).toInt() / 1000.0
}

val maxPaths = 10000

fun Aggregate<Int>.bullsEye(metric: Field<Int, Double>): Double {
    // Creates a gradient from a randomly chosen node (using gossipMin), measuring distances based on the provided metric.
    val distToRandom = distanceTo(gossipMin(localId) == localId, metric, maxPaths, isRiemannianManifold = false)

    // Finds the node that is farthest from the random starting node. This will serve as the first “extreme” of the network.
    val firstExtreme = gossipMax(distToRandom to localId, compareBy { it.first }).second

    // Builds a distance gradient starting from the first extreme node.
    val distanceToExtreme = distanceTo(firstExtreme == localId, metric, maxPaths)
//    return distanceToExtreme

    // Finds the node that is farthest from the first extreme.
    // This defines the other end of the main axis (the second “extreme”).
    val (distanceBetweenExtremes, otherExtreme) =
        gossipMax(distanceToExtreme to localId, compareBy { it.first })

    // Builds a distance gradient from the second extreme.
    val distanceToOtherExtreme = distanceTo(otherExtreme == localId, metric, maxPaths)

    // Approximates the center of the network by computing the intersection of diagonals between the two extremes,
    // and finds the closest node to that point.
    val distanceFromMainDiameter = abs(distanceBetweenExtremes - distanceToExtreme - distanceToOtherExtreme)
    val distanceFromOpposedDiagonal = abs(distanceToExtreme - distanceToOtherExtreme)
    val approximateDistance = hypot(distanceFromOpposedDiagonal, distanceFromMainDiameter)
    val centralNode = gossipMin(approximateDistance to localId, compareBy { it.first }).second

    // Measures how far each node is from the computed center.
    val distanceFromCenter = distanceTo(centralNode == localId, metric, maxPaths)
//    return distanceFromCenter
    return when (distanceFromCenter) {
        in 0.0..1.0 -> 25.0
        in 1.0..4.0 -> 75.0
        in 4.0..7.0 -> 50.0
        in 7.0..10.0 -> 0.0
        else -> 85.0
    }
}