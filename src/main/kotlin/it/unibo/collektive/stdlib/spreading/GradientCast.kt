/*
 * Copyright (c) 2025, Danilo Pianini, Nicolas Farabegoli, Elisa Tronetti,
 * and all authors listed in the `build.gradle.kts` and the generated `pom.xml` file.
 *
 * This file is part of Collektive, and is distributed under the terms of the Apache License 2.0,
 * as described in the LICENSE file in this project's repository's top directory.
 */

package it.unibo.collektive.stdlib.spreading

import it.unibo.collektive.aggregate.Field
import it.unibo.collektive.aggregate.api.*
import it.unibo.collektive.stdlib.fields.minValueBy
import it.unibo.collektive.stdlib.util.Reducer
import it.unibo.collektive.stdlib.util.coerceIn
import it.unibo.collektive.stdlib.util.hops
import it.unibo.collektive.stdlib.util.nonOverflowingPlus
import kotlin.contracts.ExperimentalContracts
import kotlin.contracts.contract

/**
 * Propagate [local] values across a spanning tree starting from the closest [source].
 *
 * If there are no sources and no neighbors, default to [local] value.
 * The [metric] function is used to compute the distance between devices in form of a field of [Distance]s.
 * [Distance]s must be in the [[bottom], [top]] range, [accumulateDistance] is used to sum distances.
 * [accumulateData] is used to modify data from neighbors on the fly, and defaults to the identity function.
 *
 * This function features *incremental repair*, and it is subject to the *rising value problem*,
 * see [Fast self-healing gradients](https://doi.org/10.1145/1363686.1364163).
 */
@JvmOverloads
inline fun <reified ID, reified Value, reified Distance> Aggregate<ID>.bellmanFordGradientCast(
    source: Boolean,
    local: Value,
    bottom: Distance,
    top: Distance,
    noinline accumulateData: (fromSource: Distance, toNeighbor: Distance, data: Value) -> Value =
        { _, _, data -> data },
    crossinline accumulateDistance: Reducer<Distance>,
    metric: Field<ID, Distance>,
): Value where ID : Any, Distance : Comparable<Distance> {
    val topValue: Pair<Distance, Value> = top to local
    val distances = metric.coerceIn(bottom, top)
    return share(topValue) { neighborData ->
        val pathsThroughNeighbors = neighborData.alignedMapValues(distances) { (fromSource, data), toNeighbor ->
            val totalDistance = accumulate(bottom, top, fromSource, toNeighbor, accumulateDistance)
            val newData = accumulateData(fromSource, toNeighbor, data)
            totalDistance to newData
        }
        val bestThroughNeighbors = pathsThroughNeighbors.minValueBy { it.value.first } ?: topValue
        when {
            source -> bottom to local
            else -> bestThroughNeighbors
        }
    }.second // return the data
}

/**
 * Propagate [local] values across a spanning tree starting from the closest [source].
 *
 * If there are no sources and no neighbors, default to [local] value.
 * The [metric] function is used to compute the distance between devices in form of a field of [Double]s,
 * [accumulateDistance] is used to sum distances.
 * [accumulateData] is used to modify data from neighbors on the fly, and defaults to the identity function.
 *
 * This function features *incremental repair*, and it is subject to the *rising value problem*,
 * see [Fast self-healing gradients](https://doi.org/10.1145/1363686.1364163).
 */
@JvmOverloads
inline fun <reified ID, reified Value> Aggregate<ID>.bellmanFordGradientCast(
    source: Boolean,
    local: Value,
    noinline accumulateData: (fromSource: Double, toNeighbor: Double, data: Value) -> Value =
        { _, _, data -> data },
    crossinline accumulateDistance: (fromSource: Double, toNeighbor: Double) -> Double = Double::plus,
    metric: Field<ID, Double>,
): Value where ID : Any =
    bellmanFordGradientCast(source, local, 0.0, Double.POSITIVE_INFINITY, accumulateData, accumulateDistance, metric)

/**
 * Propagate [local] values across multiple spanning trees starting from all the devices in which [source] holds,
 * retaining the value of the closest source.
 *
 * If there are no sources, default to [local] value.
 * The [metric] function is used to compute the distance between devices in form of a field of [Distance]s.
 * [Distance]s must be in the [[bottom], [top]] range, [accumulateDistance] is used to sum distances.
 * [accumulateData] is used to modify data from neighbors on the fly, and defaults to the identity function.
 *
 * This function features *fast repair*, and it is **not** subject to the *rising value problem*,
 * see [Fast self-healing gradients](https://doi.org/10.1145/1363686.1364163).
 *
 * On the other hand, it requires larger messages and more processing than the classic [bellmanFordGradientCast].
 * Performance can be optimized if an upper bound on the network diameter is provided through [maxDiameter].
 */
@OptIn(DelicateCollektiveApi::class)
@JvmOverloads
inline fun <reified ID : Any, reified Value, reified Distance : Comparable<Distance>> Aggregate<ID>.gradientCast(
    source: Boolean,
    local: Value,
    bottom: Distance,
    top: Distance,
    metric: Field<ID, Distance>,
    maxDiameter: Int = Integer.MAX_VALUE,
    noinline accumulateData: (fromSource: Distance, toNeighbor: Distance, neighborData: Value) -> Value =
        { _, _, data -> data },
    crossinline accumulateDistance: Reducer<Distance>,
): Value {
    val coercedMetric = metric.coerceIn(bottom, top)
    val fromLocalSource = if (source) GradientPath(bottom, emptyList<ID>(), local) else null
    return exchange(fromLocalSource) { neighborData: Field<ID, GradientPath<ID, Value, Distance>?> ->
        val neighbors = neighborData.neighbors
        // Accumulated distances with neighbors, to be used to exclude invalid paths
        val accDistances = neighborData.alignedMapValues(coercedMetric) { path, distance ->
            path?.distance?.let { accumulateDistance(it, distance) }
        }
        val neighborAccumulatedDistances = accDistances.excludeSelf()
        val nonLoopingPaths = neighborData.alignedMap(accDistances, coercedMetric) { id, path, accDist, distance ->
            when {
                id == localId || path == null || path.length > maxDiameter || localId in path.hops-> null
                // Remove paths that go through a neighbor along a path that is not shorter than a direct connection.
                accDist != null && path.hops.asSequence()
                    .filter { it in neighbors }
                    .map { neighborAccumulatedDistances[it] }
                    .any { it == null || it < accDist } -> null
                // Transform the remaining paths
                else -> accDist to lazy {
                    path.update(id, distance, bottom, top, accumulateDistance, accumulateData)
                }
            }
        }.excludeSelf().values.asSequence().filterNotNull().sortedBy { it.first }.map { it.second.value }
        val best = when {
            fromLocalSource != null -> sequenceOf(fromLocalSource)
            else -> {
                val pathsHopSets by lazy { nonLoopingPaths.associate { it.nextHop to it.hops.toSet() } }
                nonLoopingPaths.filter { reference ->
                    /*
                     * Path-coherence: paths that contain inconsistent information must be removed.
                     * In particular, if some path passes through A and then B, and another reaches the source
                     * through B and then A, we must keep only the shortest
                     * (unless they have the same path-length, namely the network is symmetric).
                     */
                    val refSize = reference.length
                    refSize <= 1 || nonLoopingPaths.all { other ->
                        val otherSize = other.length
                        val otherIsShorter = otherSize < refSize
                        when {
                            otherIsShorter || otherSize == refSize && other.distance != reference.distance -> {
                                // these are ordered the same as reference.hops
                                val otherHops = pathsHopSets[other.nextHop].orEmpty()
                                val commonHops = reference.hops.filter { it in otherHops }
                                when (commonHops.size) {
                                    0, 1 -> true
                                    else -> {
                                        // otherHops and commonHops must have the same order for all elements in commonHops
                                        val commonIterator = commonHops.iterator()
                                        val otherIterator = otherHops.iterator()
                                        var matches = 0
                                        while (commonIterator.hasNext() && otherIterator.hasNext()) {
                                            val common = commonIterator.next()
                                            val matchesSoFar = matches
                                            while (otherIterator.hasNext() && matchesSoFar == matches) {
                                                if (common == otherIterator.next()) {
                                                    matches++
                                                }
                                            }
                                        }
                                        matches == commonHops.size
                                    }
                                }
                            }
                            else -> true
                        }
                    }
                }
            }
        }
        val bestLazyList = best.map { lazy { it } }.toList()
        mapNeighborhood { neighbor -> bestLazyList.firstOrNull { it.value.hops.lastOrNull() != neighbor }?.value }
    }.local.value?.data ?: local
}

/**
 * Propagate [local] values across multiple spanning trees starting from all the devices in which [source] holds,
 * retaining the value of the closest source.
 *
 * If there are no sources, default to [local] value.
 * The [metric] function is used to compute the distance between devices in form of a field of [Double]s.
 * [accumulateData] is used to modify data from neighbors on the fly, and defaults to the identity function.
 *
 * This function features *fast repair*, and it is **not** subject to the *rising value problem*,
 * see [Fast self-healing gradients](https://doi.org/10.1145/1363686.1364163).
 *
 * On the other hand, it requires larger messages and more processing than the classic [bellmanFordGradientCast].
 * Performance can be optimized if an upper bound on the network diameter is provided through [maxDiameter].
 */
@JvmOverloads
inline fun <reified ID : Any, reified Type> Aggregate<ID>.gradientCast(
    source: Boolean,
    local: Type,
    metric: Field<ID, Double>,
    maxDiameter: Int = Integer.MAX_VALUE,
    noinline accumulateData: (fromSource: Double, toNeighbor: Double, data: Type) -> Type = { _, _, data -> data },
    crossinline accumulateDistance: Reducer<Double> = Double::plus,
): Type = gradientCast(
    source,
    local,
    0.0,
    Double.POSITIVE_INFINITY,
    metric,
    maxDiameter,
    accumulateData,
    accumulateDistance,
)

/**
 * Propagate [local] values across multiple spanning trees starting from all the devices in which [source] holds,
 * retaining the value of the closest source.
 *
 * If there are no sources, default to [local] value.
 * The [metric] function is used to compute the distance between devices in form of a field of [Int]s,
 * [accumulateDistance] is used to accumulate distances, defaulting to a plain sum.
 * [accumulateData] is used to modify data from neighbors on the fly, and defaults to the identity function.
 *
 * This function features *fast repair*, and it is **not** subject to the *rising value problem*,
 * see [Fast self-healing gradients](https://doi.org/10.1145/1363686.1364163).
 *
 * On the other hand, it requires larger messages and more processing than the classic [bellmanFordGradientCast].
 * Performance can be optimized if an upper bound on the network diameter is provided through [maxDiameter].
 */
@JvmOverloads
inline fun <reified ID : Any, reified Type> Aggregate<ID>.intGradientCast(
    source: Boolean,
    local: Type,
    metric: Field<ID, Int>,
    maxDiameter: Int = Integer.MAX_VALUE,
    noinline accumulateData: (fromSource: Int, toNeighbor: Int, data: Type) -> Type = { _, _, data -> data },
    crossinline accumulateDistance: Reducer<Int> = Int::nonOverflowingPlus,
): Type = gradientCast(
    source,
    local,
    0,
    Int.MAX_VALUE,
    metric,
    maxDiameter,
    accumulateData,
    accumulateDistance,
)

/**
 * Propagate [local] values across multiple spanning trees starting from all the devices in which [source] holds,
 * retaining the value of the closest source, using the hop count as distance metric.
 *
 * If there are no sources, default to [local] value.
 * [accumulateData] is used to modify data from neighbors on the fly, and defaults to the identity function.
 *
 * This function features *fast repair*, and it is **not** subject to the *rising value problem*,
 * see [Fast self-healing gradients](https://doi.org/10.1145/1363686.1364163).
 *
 * On the other hand, it requires larger messages and more processing than the classic [bellmanFordGradientCast].
 * Performance can be optimized if an upper bound on the network diameter is provided through [maxDiameter].
 */
@JvmOverloads
inline fun <reified ID : Any, reified Type> Aggregate<ID>.hopGradientCast(
    source: Boolean,
    local: Type,
    maxDiameter: Int = Integer.MAX_VALUE,
    noinline accumulateData: (fromSource: Int, toNeighbor: Int, data: Type) -> Type = { _, _, data -> data },
): Type = intGradientCast(source, local, hops(), maxDiameter, accumulateData, Int::plus)

/**
 * Provided a list of [sources], propagates information from each, collecting it in a map.
 *
 * If there are no sources and no neighbors, default to [local] value.
 * The [metric] function is used to compute the distance between devices in form of a field of [Float]s.
 * [accumulateData] is used to modify data from neighbors on the fly, and defaults to the identity function.
 */
@JvmOverloads
inline fun <reified ID : Any, reified Value, reified Distance : Comparable<Distance>> Aggregate<ID>.multiGradientCast(
    sources: Iterable<ID>,
    local: Value,
    bottom: Distance,
    top: Distance,
    metric: Field<ID, Distance>,
    maxDiameter: Int = Integer.MAX_VALUE,
    noinline accumulateData: (fromSource: Distance, toNeighbor: Distance, data: Value) -> Value =
        { _, _, data -> data },
    crossinline accumulateDistance: Reducer<Distance>,
): Map<ID, Value> = sources.associateWith { source ->
    alignedOn(source) {
        gradientCast(
            source == localId,
            local,
            bottom,
            top,
            metric,
            maxDiameter,
            accumulateData,
            accumulateDistance,
        )
    }
}

/**
 * Provided a list of [sources], propagates information from each, collecting it in a map.
 *
 * If there are no sources and no neighbors, default to [local] value.
 * The [metric] function is used to compute the distance between devices in form of a field of [Float]s.
 * [accumulateData] is used to modify data from neighbors on the fly, and defaults to the identity function.
 */
@JvmOverloads
inline fun <reified ID : Any, reified Value> Aggregate<ID>.multiGradientCast(
    sources: Iterable<ID>,
    local: Value,
    metric: Field<ID, Double>,
    maxDiameter: Int = Integer.MAX_VALUE,
    noinline accumulateData: (fromSource: Double, toNeighbor: Double, data: Value) -> Value = { _, _, data -> data },
): Map<ID, Value> = sources.associateWith { source ->
    alignedOn(source) {
        gradientCast(
            source = source == localId,
            local = local,
            metric = metric,
            maxDiameter = maxDiameter,
            accumulateData,
            Double::plus,
        )
    }
}

/**
 * Provided a list of [sources], propagates information from each, collecting it in a map.
 *
 * If there are no sources and no neighbors, default to [local] value.
 * The [metric] function is used to compute the distance between devices in form of a field of [Float]s.
 * [accumulateData] is used to modify data from neighbors on the fly, and defaults to the identity function.
 */
@JvmOverloads
inline fun <reified ID : Any, reified Value> Aggregate<ID>.multiIntGradientCast(
    sources: Iterable<ID>,
    local: Value,
    metric: Field<ID, Int> = hops(),
    maxDiameter: Int = Integer.MAX_VALUE,
    noinline accumulateData: (fromSource: Int, toNeighbor: Int, data: Value) -> Value = { _, _, data -> data },
): Map<ID, Value> = sources.associateWith { source ->
    alignedOn(source) {
        intGradientCast(
            source = source == localId,
            local = local,
            metric = metric,
            maxDiameter = maxDiameter,
            accumulateData = accumulateData,
            accumulateDistance = Int::nonOverflowingPlus,
        )
    }
}

/**
 * A path segment along a potential field that reaches the current device,
 * after [distance], starting from [source],
 * carrying [data] through the [nextHop].
 *
 * This data class is designed to be shared within [gradientCast] and derivative functions.
 */
data class GradientPath<ID: Any, Value, Distance: Comparable<Distance>> (
    val distance: Distance,
    val hops: List<ID>,
    val data: Value,
): Comparable<GradientPath<ID, Value, Distance>> {

    val source: ID get() = hops.first()

    val nextHop get() = hops.last()

    val length get() = hops.size

    operator fun contains(id: ID): Boolean = hops.contains(id)

    /**
     * Updates this path adding information about the local device.
     */
    @OptIn(DelicateCollektiveApi::class)
    inline fun update(
        neighbor: ID,
        distanceToNeighbor: Distance,
        bottom: Distance,
        top: Distance,
        crossinline accumulateDistance: Reducer<Distance>,
        crossinline accumulateData: (fromSource: Distance, toNeighbor: Distance, data: Value) -> Value
    ): GradientPath<ID, Value, Distance> {
        val totalDistance = accumulate(bottom, top, distance, distanceToNeighbor, accumulateDistance)
        val updatedData = accumulateData(distance, distanceToNeighbor, data)
        return GradientPath(totalDistance, hops + neighbor, updatedData)
    }

    override fun compareTo(other: GradientPath<ID, Value, Distance>) =
        compareBy<GradientPath<ID, Value, Distance>> { it.distance }.compare(this, other)
}

@OptIn(ExperimentalContracts::class)
inline fun <D: Comparable<D>> accumulate(
    bottom: D,
    top: D,
    distance: D,
    distanceToNeighbor: D,
    accumulator: Reducer<D>
): D {
    contract {
        callsInPlace(accumulator, kotlin.contracts.InvocationKind.EXACTLY_ONCE)
    }
    val totalDistance = accumulator(distance, distanceToNeighbor).coerceIn(bottom, top)
    check(totalDistance >= distance && totalDistance >= distanceToNeighbor) {
        "The provided distance accumulation function violates the triangle inequality: " +
                "accumulating $distance and $distanceToNeighbor produced $totalDistance"
    }
    return totalDistance
}
