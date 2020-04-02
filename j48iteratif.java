package crossValidation;
/* In this code block, testing was performed on the marble data set of the J48 decision
 * tree using the weka library. The J48 classification success of the Coinfidence factor
 * has been tested using 10-Fold cross-validation.
 * */
import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class j48iteratif {
	
	public static double round(double value, int places) {
		if (places < 0) throw new IllegalArgumentException();
		long factor = (long) Math.pow(10, places);
		value = value * factor;
		long tmp = Math.round(value);
		return (double) tmp / factor;
	}
	//Array average function
	public static double calculateAverage(Double dizi[]){ 
		double toplam = 0; 
		for (int i = 0; i < dizi.length; i++){ 
			toplam = toplam + dizi[i]; 					
		} 
		return (double) (toplam / dizi.length); 	
	} 

	public static double calculateStandartDeviation(Double dizi[], double ort) { 
		double kareToplam = 0; 
		for (int i = 0; i < dizi.length; i++){ 
			kareToplam = kareToplam + dizi[i] * dizi[i]; 			
		} 
		return (double) Math.sqrt(kareToplam / dizi.length - ort*ort);	
	} 
	
	public static void main(String[] args) throws Exception {
		// loads data and set class index
		Instances data = DataSource.read("path\\mix4sHistLBP.arff");
		data.setClassIndex(data.numAttributes()-1);
		String crs="";	
		J48 j48tree = new J48();
	    j48tree.setNumDecimalPlaces(3);
		j48tree.setSubtreeRaising(true);
		j48tree.setMinNumObj(1);
		j48tree.setDoNotCheckCapabilities(false);
		j48tree.setUnpruned(true);
		
		int seed = 1;
		int folds = 10;
		
		for (int m = 1; m <10; m++) {	
			
			j48tree.setConfidenceFactor((float) (0.1*m));
			// randomize data
			Random rand = new Random(seed);
			Instances randData = new Instances(data);
			randData.randomize(rand);
			if (randData.classAttribute().isNominal())
				randData.stratify(folds);
	
			Double std[]=new Double[folds];
				
			Evaluation evalAll = new Evaluation(randData);
			for (int n = 0; n < folds; n++) {
				Evaluation eval = new Evaluation(randData);
				Instances train = randData.trainCV(folds, n);
				Instances test = randData.testCV(folds, n);
				
				j48tree.buildClassifier(train);
				eval.evaluateModel(j48tree, test);
				evalAll.evaluateModel(j48tree, test);	
				std[n]=(eval.correct()/(eval.correct()+eval.incorrect()))*100;			
			}
			Double accuracy=evalAll.correct()/(evalAll.correct()+evalAll.incorrect());
			crs=crs+round((accuracy*100),2)+",";	
			
			// output evaluation
			System.out.println(j48tree.getConfidenceFactor());			
		}
		System.out.println("10-Fold Cross Validation:"+crs);
	}
	
	
}
