/*
 * Copyright (c) 2025, Danilo Pianini, Nicolas Farabegoli, Elisa Tronetti,
 * and all authors listed in the `build.gradle.kts` and the generated `pom.xml` file.
 *
 * This file is part of Collektive, and is distributed under the terms of the Apache License 2.0,
 * as described in the LICENSE file in this project's repository's top directory.
 */

package it.unibo.collektive.stdlib.spreading

import it.unibo.collektive.aggregate.Field
import it.unibo.collektive.aggregate.api.Aggregate
import it.unibo.collektive.aggregate.api.DelicateCollektiveApi
import it.unibo.collektive.aggregate.api.exchange
import it.unibo.collektive.aggregate.api.mapNeighborhood
import it.unibo.collektive.aggregate.api.share
import it.unibo.collektive.stdlib.fields.minValueBy
import it.unibo.collektive.stdlib.util.Reducer
import it.unibo.collektive.stdlib.util.coerceIn
import it.unibo.collektive.stdlib.util.hops
import it.unibo.collektive.stdlib.util.nonOverflowingPlus
import kotlin.jvm.JvmOverloads

const val DEFAULT_MAX_PATHS = 2

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
            val totalDistance = accumulateDistance(fromSource, toNeighbor).coerceIn(bottom, top)
            check(totalDistance >= fromSource && totalDistance >= toNeighbor) {
                "The provided distance accumulation function violates the triangle inequality: " +
                        "accumulating $fromSource and $toNeighbor produced $totalDistance"
            }
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
 * On the other hand, it requires larger messages and more processing than the classic
 * [bellmanFordGradientCast].
 */
@OptIn(DelicateCollektiveApi::class)
@JvmOverloads
inline fun <reified ID : Any, reified Value, reified Distance : Comparable<Distance>> Aggregate<ID>.gradientCast(
    source: Boolean,
    local: Value,
    bottom: Distance,
    top: Distance,
    metric: Field<ID, Distance>,
    maxPaths: Int = DEFAULT_MAX_PATHS,
    isRiemannianManifold: Boolean = true,
    noinline accumulateData: (fromSource: Distance, toNeighbor: Distance, neighborData: Value) -> Value =
        { _, _, data -> data },
    crossinline accumulateDistance: Reducer<Distance>,
): Value {
    require(maxPaths > 0) {
        "Computing the gradient requires at least one-path memory"
    }
    val coercedMetric = metric.coerceIn(bottom, top)
    val fromLocalSource = if (source) setOf(GradientPath(bottom, emptyList<ID>(), local)) else emptySet()
    return exchange(fromLocalSource) { neighborData: Field<ID, Set<GradientPath<ID, Value, Distance>>> ->
        val nonLoopingPaths = neighborData.alignedMap(metric) { id, paths, distance ->
            when (id) {
                localId -> emptyList()
                else -> paths.filter { localId !in it.hops }
                    .map { it.update(id, distance, bottom, top, accumulateDistance, accumulateData) }
            }
        }.excludeSelf().values.flatten().sorted()
        val allNeighbors = neighborData.neighbors
        val shortestBySource = mutableMapOf<ID, GradientPath<ID, Value, Distance>>()
        val shortestByNextHop = mutableMapOf<ID, GradientPath<ID, Value, Distance>>()
        fun MutableMap<ID, *>.removeIfSame(key: ID, current: Any, other: Any) {
            if (current == other) {
                remove(key)
            }
        }
        nonLoopingPaths.forEach { reference ->
            val source = reference.source
            val bestBySource = shortestBySource.getOrPut(source) { reference }
            if (bestBySource == reference || bestBySource.distance > reference.distance) {
                val nextHop = checkNotNull(reference.nextHop)
                val bestByNextHop = shortestByNextHop.getOrPut(nextHop) { reference }
                if (bestByNextHop == reference || bestByNextHop.distance > reference.distance) {
                    /*
                     * Path-coherence: paths that contain inconsistent information must be removed.
                     * In particular, if some path passes through A and then B, and another reaches
                     * through B and then A, we must keep only the shortest
                     * (unless they have the same path-length, namely the network is symmetric).
                     */
                    val refSize = reference.hops.size
                    val incoherent = refSize > 1 && nonLoopingPaths.any { other ->
                        val otherSize = other.hops.size
                        var coherent = reference.hops.subList(0, refSize - 1).asReversed().asSequence()
                            .filter { it in allNeighbors }
                            .all {
                                // Always false for Riemannian manifolds
                                val distanceOfDirectConnection = shortestByNextHop[it]?.distance ?: top
                                reference.distance < distanceOfDirectConnection
                            }
                        val shorter = coherent && otherSize < refSize
                        if (shorter || otherSize == refSize && other.distance != reference.distance) {
                            var hopIndex = 0
                            while (coherent && hopIndex < otherSize) {
                                val hop = other.hops[hopIndex]
                                val indexOfHop = reference.hops.indexOf(hop)
                                if (indexOfHop >= 0) {
                                    val subPathForward = other.hops.subList(hopIndex, otherSize)
                                    coherent = reference.hops.subList(0, indexOfHop).none { it in subPathForward }
                                }
                                hopIndex++
                            }
                        }
                        !coherent
                    }
                    if (incoherent) {
                        shortestBySource.removeIfSame(source, bestBySource, reference)
                        shortestByNextHop.removeIfSame(nextHop, bestByNextHop, reference)
                    } else {
                        shortestByNextHop[nextHop] = reference
                        shortestBySource[source] = reference
                    }
                } else {
                    shortestBySource.removeIfSame(source, bestBySource, reference)
                }
            }
        }
        val relevantPaths = shortestBySource.values.intersect(shortestByNextHop.values).sorted().asSequence()
        /*
         * Keep at most maxPaths paths, including the local source.
         */
        val maxIndirectPaths = maxPaths - fromLocalSource.size
        mapNeighborhood { neighbor ->
            fromLocalSource + relevantPaths.filter { it.nextHop != neighbor }.take(maxIndirectPaths)
        }
    }.local.value.firstOrNull()?.data ?: local
}

/**
 * Propagate [local] values across multiple spanning trees starting from all the devices in which [source] holds,
 * retaining the value of the closest source.
 *
 * If there are no sources, default to [local] value.
 * The [metric] function is used to compute the distance between devices in form of a field of [Double]s,
 * [accumulateDistance] is used to accumulate distances, defaulting to a plain sum.
 * [accumulateData] is used to modify data from neighbors on the fly, and defaults to the identity function.
 *
 * This function features *fast repair*, and it is **not** subject to the *rising value problem*,
 * see [Fast self-healing gradients](https://doi.org/10.1145/1363686.1364163).
 *
 * On the other hand, it requires larger messages and more processing than the classic
 * [bellmanFordGradientCast].
 */
@JvmOverloads
inline fun <reified ID : Any, reified Type> Aggregate<ID>.gradientCast(
    source: Boolean,
    local: Type,
    metric: Field<ID, Double>,
    maxPaths: Int = DEFAULT_MAX_PATHS,
    isRiemannianManifold: Boolean = true,
    noinline accumulateData: (fromSource: Double, toNeighbor: Double, data: Type) -> Type = { _, _, data -> data },
    crossinline accumulateDistance: Reducer<Double> = Double::plus,
): Type = gradientCast(
    source,
    local,
    0.0,
    Double.POSITIVE_INFINITY,
    metric,
    maxPaths,
    isRiemannianManifold,
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
 * On the other hand, it requires larger messages and more processing than the classic
 * [bellmanFordGradientCast].
 */
@JvmOverloads
inline fun <reified ID : Any, reified Type> Aggregate<ID>.intGradientCast(
    source: Boolean,
    local: Type,
    metric: Field<ID, Int>,
    maxPaths: Int = DEFAULT_MAX_PATHS,
    isRiemannianManifold: Boolean = true,
    noinline accumulateData: (fromSource: Int, toNeighbor: Int, data: Type) -> Type = { _, _, data -> data },
    crossinline accumulateDistance: Reducer<Int> = Int::nonOverflowingPlus,
): Type = gradientCast(
    source,
    local,
    0,
    Int.MAX_VALUE,
    metric,
    maxPaths,
    isRiemannianManifold,
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
 * On the other hand, it requires larger messages and more processing than the classic
 * [bellmanFordGradientCast].
 */
@JvmOverloads
inline fun <reified ID : Any, reified Type> Aggregate<ID>.hopGradientCast(
    source: Boolean,
    local: Type,
    maxPaths: Int = DEFAULT_MAX_PATHS,
    noinline accumulateData: (fromSource: Int, toNeighbor: Int, data: Type) -> Type = { _, _, data -> data },
): Type = intGradientCast(source, local, hops(), maxPaths, true, accumulateData, Int::plus)

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
    maxPaths: Int = DEFAULT_MAX_PATHS,
    isRiemannianManifold: Boolean = true,
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
            maxPaths,
            isRiemannianManifold,
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
    maxPaths: Int = DEFAULT_MAX_PATHS,
    isRiemannianManifold: Boolean = true,
    noinline accumulateData: (fromSource: Double, toNeighbor: Double, data: Value) -> Value = { _, _, data -> data },
): Map<ID, Value> = sources.associateWith { source ->
    alignedOn(source) {
        gradientCast(
            source = source == localId,
            local = local,
            metric = metric,
            maxPaths = maxPaths,
            isRiemannianManifold = isRiemannianManifold,
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
    maxPaths: Int = DEFAULT_MAX_PATHS,
    isRiemannianManifold: Boolean = true,
    noinline accumulateData: (fromSource: Int, toNeighbor: Int, data: Value) -> Value = { _, _, data -> data },
): Map<ID, Value> = sources.associateWith { source ->
    alignedOn(source) {
        intGradientCast(
            source = source == localId,
            local = local,
            metric = metric,
            maxPaths = maxPaths,
            isRiemannianManifold = isRiemannianManifold,
            accumulateData = accumulateData,
            accumulateDistance = Int::nonOverflowingPlus,
        )
    }
}

/**
 * A path segment along a potential field that reaches the current device,
 * after [distance], starting from [source],
 * passing [through] an intermediate direct neighbor,
 * carrying [data].
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

    /**
     * Returns `true` if this path has been directly provided by a source
     * (namely, [source] == [through]).
     */
    val comesFromSource get() = hops.isEmpty()

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
        val totalDistance = accumulateDistance(distance, distanceToNeighbor).coerceIn(bottom, top)
        check(totalDistance >= distance && totalDistance >= distanceToNeighbor) {
            "The provided distance accumulation function violates the triangle inequality: " +
                    "accumulating $distance and $distanceToNeighbor produced $totalDistance"
        }
        val updatedData = accumulateData(distance, distanceToNeighbor, data)
        return GradientPath(totalDistance, hops + neighbor, updatedData)
    }

    override fun compareTo(other: GradientPath<ID, Value, Distance>) =
        compareBy<GradientPath<ID, Value, Distance>> { it.distance }.compare(this, other)
}
