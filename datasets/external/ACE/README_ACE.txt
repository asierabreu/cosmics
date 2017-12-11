DATA SITE: ftp://sohoftp.nascom.nasa.gov/sdb/goes/ace/daily/

FILE FORMAT:

# Units: proton flux p/cs2-sec-ster
# Status(S): 0 = nominal data, 1 to 8 = bad data record, 9 = no data
# Missing data values: -1.00e+05
# Source: ACE Satellite - Solar Isotope Spectrometer
#
# 5-minute averaged Real-time Integral Flux of High-energy Solar Protons
# 
#                 Modified Seconds
# UT Date   Time   Julian  of the      ---- Integral Proton Flux ----
# YR MO DA  HHMM     Day     Day       S    > 10 MeV    S    > 30 MeV
#--------------------------------------------------------------------


RETRIEVE COMMAND:

export FTP="ftp://sohoftp.nascom.nasa.gov/sdb/goes/ace/daily/"
export PRE=201409
export SUF=_ace_sis_5m.txt
for i in {1..30};do data=$FTP$PRE$(printf "%02d" ${i})$SUF;wget $data;done

MERGE FILES COMMAND:

cat *_sis_5m.txt | awk '{print $1"-"$2"-"$3"T"$4 " " $8 " " $10}' | grep -v "#" | grep -v ":" >> Ace_2014_01.txt

CLEANUP:

rm *_sis_5m.txt
