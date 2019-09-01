package dl.network

class MLPNetwork {

  private var iterNum:Long = 0
  private var batSize:Int = 0
  private var learnRate:Double = 0.01

  def iterNumber() = iterNum
  def iterNumber(num:Long) = {
    iterNum = num
    this
  }

  def batchSize() = batSize
  def batchSize(size:Int) = {
    batSize = size
    this
  }

  def learningRate() = learnRate
  def learningRate(rate:Double) = {
    learnRate = rate
    this
  }

  def layer() = {

    this
  }



}
