for sys in Pn0 Pn60 Pn120 Pn180 Hn0 Hn180 Hn60 Hn120
do
    for d in 10 5 4 2 1
    do
        dir="last_10ns/${sys}_l10.gro"
        if [ ${sys#?n} = 0 ]
        then 
            echo Running jMa_scaled on $sys with d = $d
            python jMa_scaled.py $dir 2 $d nats_smr/rand/d${d}$sys

            echo Running jMa_smr on $sys with d = $d
            python jMa_smr.py $dir 2 $d nats_smr/drg/d${d}$sys


        else
            echo Running jMa_scaled on $sys with d = $d
            python jMa_scaled.py $dir 3 $d nats_smr/rand/d${d}$sys

            echo Running jMa_smr on $sys with d = $d
            python jMa_smr.py $dir 3 $d nats_smr/drg/d${d}$sys

            echo Running jMa_smr2 on $sys with d = $d
            python jMa_smr2.py $dir 3 $d nats_smr/dbsa/d${d}$sys
        fi
        echo system $sys with d = $d is done!
    done
done