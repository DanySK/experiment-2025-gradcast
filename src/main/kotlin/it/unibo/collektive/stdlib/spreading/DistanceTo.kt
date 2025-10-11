package it.unibo.collektive.stdlib.spreading

import it.unibo.collektive.aggregate.Field
import it.unibo.collektive.aggregate.api.Aggregate
import it.unibo.collektive.stdlib.ints.FieldedInts.toDouble
import it.unibo.collektive.stdlib.util.Reducer
import it.unibo.collektive.stdlib.util.hops
import kotlin.jvm.JvmOverloads

/**
 * Compute the distance from the closest [source], using [Double]s.
 *
 * The distance between neighboring devices is computed using the [metric] function,
 * and defaults to the hop distance.
 */
@JvmOverloads
inline fun <reified ID : Any> Aggregate<ID>.distanceTo(
    source: Boolean,
    metric: Field<ID, Double> = hops().toDouble(),
    maxDiameter: Int = Integer.MAX_VALUE,
): Double = distanceTo(source, 0.0, Double.POSITIVE_INFINITY, metric, maxDiameter, Double::plus)

/**
 * Compute the [Distance] from the closest [source], starting from [bottom] and up to [top].
 *
 * the [Distance] between neighboring devices is computed using the [metric] function,
 * the distance summation is governed by the [accumulateDistance] function.
 */
inline fun <reified ID : Any, reified Distance : Comparable<Distance>> Aggregate<ID>.distanceTo(
    source: Boolean,
    bottom: Distance,
    top: Distance,
    metric: Field<ID, Distance>,
    maxDiameter: Int = Integer.MAX_VALUE,
    noinline accumulateDistance: Reducer<Distance>,
): Distance = gradientCast(
    source = source,
    local = if (source) bottom else top,
    bottom = bottom,
    top = top,
    metric = metric,
    maxDiameter = maxDiameter,
    accumulateData = { neighborToSource, hereToNeighbor, _ ->
        accumulateDistance(neighborToSource, hereToNeighbor)
    },
    accumulateDistance = accumulateDistance,
)

/**
 * Computes the hop distance from the closest [source].
 */
inline fun <reified ID : Any> Aggregate<ID>.hopDistanceTo(source: Boolean): Int =
    hopGradientCast(source = source, local = 0) { neighborToSource, hereToNeighbor, _ ->
        neighborToSource + hereToNeighbor
    }
