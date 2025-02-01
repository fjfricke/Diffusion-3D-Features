NOTICE
======

This archive contains ground-truth data for high resolution models with 100k polygons. A version for the low resolution models is also available at http://robertodyke.com/shrec2020/


ABOUT
=====

This is an archive of the ground-truths for the benchmark dataset for SHREC'20 Track: Shape Correspondence with Non-Isometric Deformations (https://doi.org/10.1016/j.cag.2020.08.008)

For more information about this track and this dataset please visit http://robertodyke.com/shrec2020/

Authors: Dyke, R. M., Lai, Y., Rosin, P. L.


GROUND-TRUTH DATA
=================

Ground-truth data is currently stored in *.mat files. Multiple useful attributes are included.

fname - The name of the original mesh
points - The indicies of locations found on the mesh (e.g., point 1 is the right eye, point 2 is the left eye, etc.). N.B. certain points like the tusks on the elephant do not map to a point on other animals, therefore these points are given a unique index, which is only used on the elephant. Please see the section below for a comprehensive list of these inconsistencies.
centroids - The medoid location of the labellers' selection.
baryc - The centroids projected onto the surface of the mesh. This is described in a conventional barycentric format: column 1 is the index of the facet the point is inside of, columns 2-4 weight the contribution of each vertex of the facet.
verts - The index of the closest vertex to each respective centroid. N.B. This is the least accurate information, it is included for convenience.

TOPOLOGICALLY INCONSISTENT POINTS
=================================

point 3 single horn tip (rhino)
point 52 left horn tip (bison/cow)
point 53 right horn tip (bison/cow)
point 54 hump tip (camels)

point 4 right nostril (elephant trunk)
point 5 left nostril (elephant trunk)
point 6 upper lip (elephant trunk)
point 55 left tusk tip (elephant)
point 56 right tusk tip (elephant)
