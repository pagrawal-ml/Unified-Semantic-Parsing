#!/bin/bash
outdir=outputs
mkdir $outdir
domainArray=("recipes" "housing" "basketball" "blocks" "calendar" "publications" "restaurants" "socialnetwork")

outdir1=${outdir}/out1
mkdir $outdir1
for domain in ${domainArray[*]}; do
     echo $domain
     python preprocess_1.py overnight-lf/ $domain $outdir1
done

java_out_dir=${outdir}/entity_normalized_out
mkdir $java_out_dir

javac OvernightEntityAnonymizer.java
java OvernightEntityAnonymizer $outdir1 $java_out_dir

final_out_dir=${outdir}/preprocessed_data
mkdir $final_out_dir
for domain in ${domainArray[*]}; do
     echo $domain
     python preprocess_2.py $java_out_dir $domain $final_out_dir
done


post_out_dir1=${outdir}/post11
mkdir $post_out_dir1
#python postprocess_tsv_2.py $final_out_dir $post_out_dir1
python postprocess_p4.py $final_out_dir $post_out_dir1


post_out_dir2=${outdir}/post2
mkdir $post_out_dir2
suffix1train=_train.tsv.entity.prune.txt
suffix1test=_test.tsv.entity.prune.txt
suffix2=.prune2.txt.orig1

for domain in ${domainArray[*]}; do
     echo $domain
     parseFile=$post_out_dir1'/'$domain$suffix1train$suffix2
     parseFile_test=$post_out_dir1'/'$domain$suffix1test$suffix2
     transformation_file="${java_out_dir}/${domain}_train.trans.txt"
     transformation_file_test="${java_out_dir}/${domain}_test.trans.txt"
     outputfile="${post_out_dir2}/${domain}_train.transout.txt"
     outputfile_test="${post_out_dir2}/${domain}_test.transout.txt"
     python entity_deanonymization.py $parseFile $transformation_file $outputfile 1
     python entity_deanonymization.py $parseFile_test $transformation_file_test $outputfile_test 1
done

post_out_dir_final=${outdir}/postprocessed_final
mkdir $post_out_dir_final
python postprocess_tsv_1.py $post_out_dir2 $post_out_dir_final
 
