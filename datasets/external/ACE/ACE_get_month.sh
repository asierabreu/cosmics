# call signature: ./ACE_get_month.sh YEAR MONTH

YEAR=$1
MONTH=$2
PRE=$(printf "%04d%02d" $YEAR $MONTH)

export FTP="ftp://sohoftp.nascom.nasa.gov/sdb/goes/ace/daily/"
export SUF=_ace_sis_5m.txt

# get data
for i in {1..31};
do 
    data=$FTP$PRE$(printf "%02d" ${i})$SUF
    wget $data
done

# merge the files
cat *_sis_5m.txt | awk '{print $1"-"$2"-"$3"T"$4 " " $8 " " $10}' | grep -v "#" | grep -v ":" >> ACE_$(printf "%04d-%02d" $YEAR $MONTH).txt

# cleanup
rm *_sis_5m.txt
