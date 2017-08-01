(ns prop.core
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.operators :as M]
            [clojure.walk :as walk]))

(defn rep [a b]
  (into [] (repeatedly a #(into [] (repeatedly b rand)))))

(defn to-vector [output-bias args entity]
  (vec
    (concat
      (for [arg args]
        (into [] (butlast (entity arg))))
       [(-> output-bias vals vec)])))

(defn nn [input output output-bias & args]
  (let [num-bias (into [] (for [arg args] (:num arg)))
        hidden-weights (->> num-bias (partition 2 1) (mapv (partial reduce rep)))
        output-weight (conj hidden-weights (rep (-> args last :num) (count output)))
        weights (cons (rep (count input) (-> args first :num)) output-weight)]
    {:input input
     :bias (to-vector output-bias args vals)
     :weights (into [] weights)
     :bias-weights (conj (vec
                           (for [arg args]
                             (vec (butlast (keys arg)))))
                         (-> output-bias keys vec))
     :output output}))

(defn net-layer [inputs weights bias bias-weights]
  (M/+
    (m/mmul
      (m/matrix [inputs])
      (m/matrix weights))
    (M/*
      (m/matrix [bias])
      (m/matrix [bias-weights]))))

(defn sigmoid [net]
  (/ 1 (+ 1 (Math/exp (- net)))))

(defn sigmoid-layer [inputs weights bias bias-weights]
  (mapv sigmoid (first (net-layer inputs weights bias bias-weights))))

(defn calc-sigmoid [{:keys [input bias weights bias-weights output]}]
  {:sig-values (loop [ind 1
                      res [(sigmoid-layer input (first weights) (first bias) (first bias-weights))]]
                 (if (>= ind (count bias-weights))
                   res
                   (recur (inc ind)
                          (conj res (sigmoid-layer (last res) (get weights ind) (get bias ind) (get bias-weights ind))))))
   :output output})

(defn indexer [vec number]
  (for [[x set1] (map-indexed vector vec)
        [y set2] (map-indexed vector set1)
        [z val] (map-indexed vector set2)
        [z val] (map-indexed vector set2)
        :when (= number val)]
    [x y z]))

(defn connector [values weights a b c]
  [(reduce get weights [a b c]) (reduce get values [a b]) (reduce get values [(+ 1 a) c])])

(defn links [{:keys [input bias weights bias-weights output] :as network}]
  (let [sig (cons input (:sig-values (calc-sigmoid network)))]
    {:weight-links (map #(connector (vec sig) weights (-> % first first) (-> % first second) (-> % first last)) (map #(indexer weights %) (flatten weights)))
     :sig sig}))

(defn connection [num links]
  (remove nil?
          (map (fn [a]
                 (when (not= () (filter #(= % num) a))
                   a))
               links)))

(defn delta-variables [weight network]
  (let [{:keys [weight-links sig]} (links network)
        output-sig (last sig)
        connect (list (connection weight weight-links))]
    {:delta-variables (if (some #(= (-> connect first first last) %) output-sig)
                        (mapv rest (first connect))
                        (loop [res (mapv rest (first connect))
                               c connect]
                          (let [new-c (first (map (fn [x]
                                                    (map (fn [cc] (let [woutput (last cc)
                                                                        woutput-links (connection woutput weight-links)]
                                                                    (filter #(= (get % 1) woutput) woutput-links)))
                                                         x))
                                                  c))
                                new-res (conj
                                          res
                                          (map (fn [x]
                                                 (map (fn [xx]
                                                        [(first xx) (last xx)])
                                                      x))
                                               new-c))]
                            (if (some #(= (-> new-c first first last) %) output-sig)
                              new-res
                              (recur new-res
                                     new-c)))))
     :last-sig output-sig}))

(defn layer-eq [[weight value]]
  (* weight (- 1 value)))

(defn output-eq [[weight value] output last-sig]
  (* weight value (- 1 value) (- value (get output (.indexOf last-sig value)))))

(defn delta-value [weight {:keys [output] :as network}]
  (let [{:keys [delta-variables last-sig]} (delta-variables weight network)
        first-vars (first delta-variables)
        last-vars (last delta-variables)
        middle-vars (butlast (rest delta-variables))
        a (layer-eq first-vars)
        b (map (fn [x] (map (fn [xx] (output-eq xx output last-sig)) x)) last-vars)
        c (map (fn [x] (map (fn [xx] (map (fn [xxx] (layer-eq xxx)) xx)) x)) middle-vars)]
    (cond
      (and (nil? middle-vars) (= first-vars last-vars)) (output-eq last-vars output last-sig)
      (nil? middle-vars) (* a (first (map #(apply + %) b)))
      :else (* a (first (map #(apply + %) (reduce (fn [acc v] (M/* v (map #(apply + %) acc))) b (reverse c))))))))

(defn new-weight [weight network learning-rate]
  (let [new-value (delta-value weight network)]
    (- weight (* learning-rate new-value))))

(defn new-network-weights [network learning-rate]
  {:new-weights (walk/postwalk (fn [x] (if (number? x)
                                         (new-weight x network learning-rate)
                                         x))
                               (:weights network))})

(defn output-error [target output-sigmoid]
  (* 0.5 (- target output-sigmoid) (- target output-sigmoid)))

(defn output-layer-error [target output-sigmoid]
  (mapv output-error target output-sigmoid))

(defn errors [{:keys [sig-values output]}]
  (let [output-errors (output-layer-error output (-> sig-values last))]
    {:output output
     :calc-output (last sig-values)
     :output-errors output-errors
     :total-error (apply + output-errors)}))

(defn train [network learning-rate num-of-iterations println?]
  (loop [x 0
         v network]
    (if (>= x num-of-iterations)
      v
      (recur (inc x)
             (let [{:keys [new-weights]} (new-network-weights v learning-rate)]
               (when println?
                 (println "Total error is: " (:total-error (-> v calc-sigmoid errors)) "Iteration is: " (inc x)))
               (assoc v :weights new-weights))))))

(defn trained-nn [network learning-rate n println?]
  (let [trained-network (train network learning-rate n println?)
        errors (-> trained-network calc-sigmoid errors)]
    (assoc errors :weights (:weights network))))
