IBR-*.txt
=========
P(referent | message) for a variety of IBR listener models. Generated using the pragmods toolkit: https://code.google.com/p/pragmods/
Format:
<problem matrix> p(r1 | m1) p(r1 | m2) p(r1 | m3) p(r2 | m1) .... p(r3 | m3)


scale_plus_6stimuli_3levels_no_fam_24_january_SCAL.csv   
======================================================
Experimental results for the "complex" experimental condition. CSV header explains format.

scales_6stimuli_3levels_no_fam_25_january_OSCA.csv
==================================================
Experimental results for the "simple" experimental condition. CSV header explains format.

facesInstances*.csv
===================
Example problem instances (problem, message, referent) for each reference level. facesInstances.csv contains all concatenated together.
Used for training the discriminative IBR models.
Format:
<problem matrix>, <message>, <referent>
