# **NURBS tutorial**

## Quick Introduction

### What are NURBS?
- NURBS stands for Non-Uniform Rational B-Splines, they are a mathematical model to represent curves and **surfaces of arbitrary shape**.

### Why does ARTIST use NURBS?
- NURBS can represent the continuous, complex and imperfect heliostat surfaces.
- NURBS can be implemented differentiable, this means they can be learned and optimized using AI algorithms.
- NURBS representing heliostat surfaces can be constructed with less parameters than constructing heliostat surfaces from point clouds or deflectometry data.
- NURBS are **precise and performant**

## Simplifications for the scope of this tutorial
- For the scope of this tutorial and for reasons of simplicity this tutorial only considers NURBS curves not surfaces.
    - The behaviour of NURBS curves in 2 dimensions can easily be transferred to NURBS surfaces in 3 dimensions.
    - Two NURBS curves can span a NURBS surface.
- NURBS are generally parametrized by functions of the form: $Q(t)=\{X(t), Y(t)\}$ Imagine t as representing the time. For a curve this could be imagined as a particle moving through space tracing out a curve. $Q(t)$ then gives the $\{x, y\}$ coordinates of the particle at time t.

## Key components and characteristics of NURBS
- NURBS are made up out of the following components: **control points, basis functions, knots, weights and degrees**
- Different manifestations of these components result in different properties of the NURBS, as for example non-uniformity or rationality.

The following will go over the components and charateristics of NURBS curves.

### Degree
- Positive integer, closely related to the order
- **Degree = Order - 1**
    - third-order curve -> represented by quadratic polynomial -> quadratic curve
- It is possible to increase the degree of the NURBS curve without changeing its form
- It is not possible to reduce the degree without changeing its form

### Control points
- Shape of the NURBS is directly determined by the control points
- More control points allows for better approximation of given curve
- Representation in the code: List of points (length of list at least: degree + 1)
- Connecting straight lines between control points form the control polygon
![](https://s3.desy.de/hackmd/uploads/0d7dc15b-c021-4fc3-b6e0-91ca73dc1c81.png)
    - Influence of the control points on the behavior of the NURBS curve
        - The only parameter that changed between the two curves is the location of control point B~7~
        - The change in the curve is limited to the local neighborhood of that control point
- Each control point influences the part of the curve nearest to it but has little or no effect on parts of the curve that are farther away
- **At any time t the particle´s position is a weighted average of all control points but the points closer to the particle are weighted more than those farther away**
- This is expressed by the following formula:
![](https://s3.desy.de/hackmd/uploads/55f9ac53-3ee6-47a7-8626-883afcaa5f39.png)
    - where k = order (degree + 1), n = number of control points, B = control points, N = basis functions

### Basis functions
- Basis functions are assigned to control points. Each control point has a corresponding basis function.
- N~ik~(t) are the basis functions, they determine how strongly control point B~i~ influences the curve at time t
- To come up with the basis functions consider the following image:
    ![](https://s3.desy.de/hackmd/uploads/336bbdc3-6841-4762-955b-05b3e8997309.png)

    - These are examplary basis functions for a NURBS curve with 5 control points. Each control point has one basis function. The red basis function is assigned to control point 2, it goes from t = 0 to t = 0.7. This is the time interval during which control point 2 controls the shape of the NURBS curve. For t = 0.8 only the basis functions of control point 3, 4 and 5 are activated thus only control points 3, 4 and 5 control the shape of the NURBS curve at that time. Since the green basis function that is assigned to control point 4 peaks at t = 0.8, this control point has the most influence on the NURBS curve at that point in time.
- Further observations:
    - At any time t, the values of all basis functions add up to exactly 1
    - At any time t, no more than k basis functions affect the curve (k = order = degree + 1), the example above is of order 3
    - A curve of order k is defined only where k of the basis functions are nonzero
- In the example above all control points affect same-sized regions of the curve and also affect the curve with the same strength, thus they are uniform (and have uniform knot vectors)
    - If this is not desired consider non-uniform NURBS and introduce non-uniform knots
- **In ARTIST only uniform NURBS are considered. The following is only explained for the sake of completeness**

### Knots
- A series of points that partition the overall time it takes the particle to move along the curve into intervals
- Representation in the code: Ordered list of numbers (length of list: degree + # of control points + 1)
- By varying the relative lengths of the intervals, the amount of time each control point affects the particle is varied (knot spans)
- Uniform knot vector = all knots are equidistant = all basis functions cover equal intervals of time
    - Example:
![](https://s3.desy.de/hackmd/uploads/71e72c57-de33-4f87-aea4-f50ca4bb73ec.png)
- Non-uniform knot vector = knot spans of different sizes = basis functions cover different intervals of time
    - Example:
![](https://s3.desy.de/hackmd/uploads/21db3e92-90c9-4043-bce6-12a2d71a05df.png)
    - Not all basis functions are the same. Some are taller and some are wider than others. This is because the knot spans vary. For smaller knot spans the basis functions become taller and narrower. For the corresponding control points, the curve is pulled more strongly to those control points.
- This consideration allows for the definition of the basis functions
![](https://s3.desy.de/hackmd/uploads/d306349d-f507-4f11-854e-89eb7c6bd6eb.png)
    - where x~i~ is the *i*th knot in the knot vector

### Knot span
- knot span of zero length: two consecutive knots have the same value (knot with **multiplicity**)
    - Full multiplicity knot: multiplicity = degree
    - Simple knot: multiplicity = 1
- If the first and last knot have full multiplicity, the NURBS curve begins and ends in a controlpoint. Full multiplicity in the first and last knot doesnt affect the uniformity property.
* **Uniformity:**
    * Example: [0,0,0,1,2,3,4,4,4]
        * knot vector starts with full multiplicity knots, followed by simple knots, ends with full multiplicity knot and the knot spand inbetween have the same distances

### Control weights
- The following explains the *rational* property. However this is not relevant for ARTIST as in ARTIST all control weights are always 1, thus they make no difference.
- If all control weights are always 1 the NURBS are non-rational which is a special subset of rational NURBS.
- Since all control points in ARTIST have a weight of 1.0 each control point has an equal influence on the shape of the curve
- Increasing the weight of one control point gives it more influence and "pulls" the curve towards that control point
- Rational curves: some or all control weights differ from 1

## NURBS in ARTIST

### General
- The NURBS in ARTIST are implemented differentiable.
- They are primarily used to model the heliostat surfaces, in the future they might also be used to model other surfaces like the receiver.
- The NURBS in ARTIST are uniform and non-rational, the name NURBS can therefore be misleading, however the uniform and non-rational implementation is a subset of NURBS.

### Code
- The NURBS code in ARTIST is contained in `nurbs.py` which can be found under `artist/util`
- `nurbs.py` implements the `NURBSSurface` class and inherits from `torch.nn.Module` allowing for gradient based calculations

### Usage
- A NURBS surface can be initialized by only providing the desired `degree_e` and `degree_n`, the `evaluation_points_e` and `evaluation_points_n` and the `control points`, where e and n stand for east and north. For a NURBS surface two degrees are necessary as two NURBS curves span the surface. Internally the uniform knot vectors `knots_e` and `knots_n` are calculated from the input.
- The user can then simply call `calculate_surface_points_and_normals()` on the `NURBSsurface` and the surface points and surface normals are calculated.
- For the calculation of the surface points and surface normals three internal steps are executed:
    1. `find_span()` is called for both directions (east and north), to determine which evaluation point corresponds to which knot in the knot vector.
    2. Next the basis functions and their derivatives are calculated, again for both directions using `basis_function_and_derivatives()`
    3. Lastly the surface points and normals are calulated from the basis functions, their derivatives and the control points.
