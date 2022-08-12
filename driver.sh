for sys in Pn0 Pn60 Pn120 Pn180 Hn0 Hn180
do
    dir="last_10ns/${sys}_l10.gro"
    if [ ${sys#?n} = 0 ]
    then 
        python jMa.py $dir 2 > nats_l10/${sys}.txt
    else
        python jMa.py $dir 3 > nats_l10/${sys}.txt
    fi
    echo system $sys is done!
done