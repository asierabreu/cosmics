DATA SITE: ftp://sohoftp.nascom.nasa.gov/sdb/goes/particle/

FILE FORMAT:

# Label: P > 1 = Particles at >1 Mev
# Label: P > 5 = Particles at >5 Mev
# Label: P >10 = Particles at >10 Mev
# Label: P >30 = Particles at >30 Mev
# Label: P >50 = Particles at >50 Mev
# Label: P>100 = Particles at >100 Mev
# Label: E>0.8 = Electrons at >0.8 Mev
# Label: E>2.0 = Electrons at >2.0 Mev
# Label: E>4.0 = Electrons at >4.0 Mev
# Units: Particles = Protons/cm2-s-sr
# Units: Electrons = Electrons/cm2-s-sr
# Source: GOES-13
# Location: W075
# Missing data: -1.00e+05
#
#                      5-minute  GOES-13 Solar Particle and Electron Flux
#
#                 Modified Seconds
# UTC Date  Time   Julian  of the
# YR MO DA  HHMM    Day     Day     P > 1     P > 5     P >10     P >30     P >50     P>100     E>0.8     E>2.0     E>4.0


RETRIEVE COMMAND:

export FTP="ftp://sohoftp.nascom.nasa.gov/sdb/goes/particle/"
export PRE=201409
export SUF=_Gp_part_5m.txt
for i in {1..30};do data=$FTP$PRE$(printf "%02d" ${i})$SUF;wget $data;done

MERGE FILES COMMAND:

cat *_Gp_part*.txt | awk '{print $1"-"$2"-"$3"T"$4 " " $7 " " $8 " " $9 " " $10 " " $11 " " $12}' | grep -v "#" | grep -v ":" >> Goes_2014_04.txt

CLEANUP:

rm *_Gp_part*.txt

