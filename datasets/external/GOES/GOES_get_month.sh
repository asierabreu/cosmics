# call signature: ./GOES_get_month.sh YEAR MONTH

YEAR=$1
MONTH=$2
PRE=$(printf "%04d%02d" $YEAR $MONTH)

export FTP="ftp://sohoftp.nascom.nasa.gov/sdb/goes/particle/"
export SUF=_Gp_part_5m.txt

# get data
for i in {1..31};
do 
    data=$FTP$PRE$(printf "%02d" ${i})$SUF
    wget $data
done

# merge the files
cat *_Gp_part*.txt | awk '{print $1"-"$2"-"$3"T"$4 " " $7 " " $8 " " $9 " " $10 " " $11 " " $12}' | grep -v "#" | grep -v ":" >> GOES_$(printf "%04d-%02d" $YEAR $MONTH).txt

# cleanup
rm *_Gp_part*.txt
