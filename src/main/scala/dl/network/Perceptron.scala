package dl.network

import breeze.linalg._


object Perceptron {

  val and = build(1,1,-1.2)
  val nand = build(-0.5,-0.5,0.7)
  val or = build(1,1,-0.5)
  val xor = (x1:Double,x2:Double)=>and(nand(x1,x2),or(x1,x2))

  def build(w1:Double,w2:Double,b:Double) = new Perceptron(w1,w2,b)

}
class Perceptron(w1:Double,w2:Double,val b:Double) {

  val w = DenseVector(w1,w2)

  def apply(x1:Double,x2:Double):Double = if(b+sum(w *:* DenseVector(x1,x2)) <= 0) 0 else 1

}