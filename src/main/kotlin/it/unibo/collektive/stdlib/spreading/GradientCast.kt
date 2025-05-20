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
import it.unibo.collektive.aggregate.api.share
import it.unibo.collektive.stdlib.fields.foldValues
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
    return share(fromLocalSource) { neighborData: Field<ID, Set<GradientPath<ID, Value, Distance>>> ->
        val nonLoopingPaths = neighborData.mapValues { paths -> paths.filter { localId !in it.hops } }
        val nonLoopingPathsMap by lazy { nonLoopingPaths.excludeSelf() }
        val neighbors by lazy { nonLoopingPaths.neighbors }
        val distanceUpdatedPaths = nonLoopingPaths.alignedMap(coercedMetric) { neighbor: ID, paths, distanceToNeighbor ->
            paths.mapNotNull { path ->
                if (path.comesFromSource) {
                    path.update(neighbor, distanceToNeighbor, bottom, top, accumulateDistance, accumulateData)
                } else if (neighbor == localId || localId in path) {
                    // Previous data, discarded anyway when reducing, or loopback to self
                    null
                } else if (isRiemannianManifold && neighbors.any { it in path }) {
                    // In Riemannian manifolds, the distance is always positive and the triangle inequality holds.
                    // Thus, we can safely discard paths that pass through a direct neighbor
                    // (except for neighbors that are sources),
                    // as the distance will be always larger than getting to the neighbor directly.
                    null
                } else {
                    // If there are neighbors along the path, make sure that these neighbors' paths to the same
                    // source do not pass through the neighbor device (loop)
                    val isValid = path.hops.all { intermediateHop ->
                        when {
                            intermediateHop !in neighbors -> true
                            else -> {
                                val relevantPath = nonLoopingPathsMap[intermediateHop]
                                    ?.firstOrNull { it.source == path.source }
                                    ?.hops
                                relevantPath != null && neighbor !in relevantPath
                            }
                        }
                    }
                    if (isValid) {
                        path.update(neighbor, distanceToNeighbor, bottom, top, accumulateDistance, accumulateData)
                    } else {
                        null
                    }
                }
            }
        }
        /*
         * Take one path per source and neighbor (the one with the shortest distance).
         */
        val candidatePaths = distanceUpdatedPaths.foldValues(
            mutableMapOf<ID, GradientPath<ID, Value, Distance>>(),
        ) { accumulator, paths ->
            paths.forEach { path ->
                val key = path.hops.first()
                val previous = accumulator.getOrPut(key) { path }
                if (previous.distance > path.distance) {
                    accumulator[key] = path
                }
            }
            accumulator
        }.values.sorted()
        /*
         * Keep at most maxPaths paths, including the local source.
         */
        val topCandidates = candidatePaths.asSequence()
            .take(maxPaths - fromLocalSource.size)
        println("$localId shares ${fromLocalSource.size + topCandidates.count()} paths")
        fromLocalSource + topCandidates
    }.firstOrNull()?.data ?: local
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

    val source: ID? get() = hops.firstOrNull()

    val nextHop get() = hops.last()

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
