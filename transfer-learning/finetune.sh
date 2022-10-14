for runseed in 0 1 2 3 4 5 6 7 8 9
do
    for dataset in 'tox21' 'hiv' 'bace' 'clintox' 'bbbp' 'sider' 'toxcast' 'muv'
    do
        python finetune.py --runseed $runseed --dataset $dataset
    done
done