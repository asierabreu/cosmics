Description : Gaia Prompt Particle Event Counts as extracted from telemetry.

File Structure:

Column 1: OBMT 		, Onboard Mission Time , units(ns) long
Column 2: OBMT_REV      , Onboard Mission Time , units(rev) float
Column 3: UTC           , Universal Time , units(s) 
Column 4: ROW           , Gaia VPU id , effectively the Sky Mapper Id, int
Column 5: RejectedPpe   , Count of Rejected PPEs, int
Column 6: Cumulative	, Cumulative Total nb. of PPEs

Note that the corresponding Field Of View (SM1 or SM2) is embedded in the file name: e.g : ASD4_Counters_FOV1_ROW1.dat

Latest Extracted at : 1st Sep 2017

FTP repository location: ftp://ftp.sciops.esa.int/pub/gaiacal
