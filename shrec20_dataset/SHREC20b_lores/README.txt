NOTICE
======

This archive contains low resolution models with 20k polygons. A high resolution archive is also available at http://robertodyke.com/shrec2020/

Please see ACKNOWLEDGEMENTS.txt to observe the attributions and licenses of the files used in this archive. Please also read the AIM@SHAPE-VISIONAIR general license included.

ABOUT
=====

This is an archive of the benchmark dataset for SHREC'20 Track: Shape Correspondence with Non-Isometric Deformations

For more information about this track and this dataset please visit http://robertodyke.com/shrec2020/

Authors: Dyke, R. M., Lai, Y., Rosin, P. L.


ARCHIVE CONTENTS
================

models/
	Contains all the shapes needed to complete this challenge 

test-sets/
	Contains five different lists of shape pairs for testing. Each pair has been selected and organised based on whether the pair contains a partial or full (i.e., watertight) scan, and the degree of non-isometry exibited.

	Note: One (or more) test sets have to be completed for submission.

	test-set0.txt - partial-to-full scans
	test-set1.txt - full-to-full highest isometry
	test-set2.txt - full-to-full high isometry
	test-set3.txt - full-to-full low isometry
	test-set4.txt - full-to-full lowest isometry

	Each test set is a comma seperated file, where each row represents a pair of shapes to be matched. Each file has two columns, the first indicates the source shape (either a partial or full scan) and the second indicates the target shape (always a full scan).

AIM@SHAPE-VISIONAIR metadata/
	Contains relevant metadata for shapes from the AIM@SHAPE & VISIONAIR Shape Repository.