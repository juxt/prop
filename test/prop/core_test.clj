;; Copyright © 2017, JUXT LTD.

(ns prop.core-test
  (:require [clojure.test :refer :all]
            [prop.core :refer :all]))

(def network (nn [1 1] [0.5 -1] {0.2 0.6 0.15 0.8} {0.05 0.1 0.15 0.2 0.25 0.3 :num 3} {0.35 0.4 0.45 0.5 :num 2}))

(def add-weights-network (assoc network :weights [[[0.11 0.22 0.33] [0.44 0.55 0.66]]
                                                  [[0.77 0.88] [0.99 0.1010] [0.1111 0.1212]]
                                                  [[0.1313 0.1414] [0.1515 0.1616]]]))

(deftest create-network
  (are [vector result] (= vector result)
       (:input add-weights-network) [1 1]
       (:output add-weights-network) [0.5 -1]
       (:bias add-weights-network) [[0.1 0.2 0.3] [0.4 0.5] [0.6 0.8]]
       (:bias-weights add-weights-network) [[0.05 0.15 0.25] [0.35 0.45] [0.2 0.15]]
       (:weights add-weights-network) [[[0.11 0.22 0.33] [0.44 0.55 0.66]]
                                       [[0.77 0.88] [0.99 0.1010] [0.1111 0.1212]]
                                       [[0.1313 0.1414] [0.1515 0.1616]]] ))

(deftest calc-sigmoid-values
  (is (= (:sig-values (calc-sigmoid add-weights-network))
         [[0.6352948493228071 0.6899744811276125 0.7436448917562621]
          [0.8013686871697223 0.7198820573000824] 
          [0.5828008483569406 0.586531846874077]])))

(deftest calc-delta-values
  (are [value result] (= value result)
       (delta-value 0.22 add-weights-network) 0.004050014136903793
       (delta-value 0.99 add-weights-network) 0.007818394737250751
       (delta-value 0.1414 add-weights-network) 0.30832931997907764))

(deftest calc-new-weights
  (are [weight result] (= weight result)
       (new-weight 0.22 add-weights-network 0.5) 0.2179749929315481
       (new-weight 0.99 add-weights-network 0.5) 0.9860908026313746
       (new-weight 0.1414 add-weights-network 0.5) -0.012764659989538824))

(deftest trained-network
  (let [trained-network (trained-nn add-weights-network 0.5 10 false)]
    (are [vector result] (= vector result)
         (:output trained-network) [0.5 -1]
         (:calc-output trained-network) [0.5590643640328539 0.18381864085211796]
         (:output-errors trained-network) [0.0017442995493027408 0.700713287214478]
         (:total-error trained-network) 0.7024575867637808
         (:weights trained-network) [[[0.11 0.22 0.33] [0.44 0.55 0.66]]
                                     [[0.77 0.88] [0.99 0.101] [0.1111 0.1212]]
                                     [[0.1313 0.1414] [0.1515 0.1616]]])))


