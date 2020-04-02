package crossValidation;
/* In this block of code, testing was performed on the marble data set of
 * the Support Vector Machine classifier known as SMO in the weka library.
 * The SMO classification success of the complexity coefficient was tested
 * using 10-fold cross-validation. In addition, NormalizedPolyKernel-PolyKernel-Puk-RBFKernel
 * can be tested separately by changing the kernel types.
 * */
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.Kernel;
import java.util.Random;

public class svmcapraz {
	
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
		for (int m = 0; m <12; m++) {
			SMO smo=new SMO();//NormalizedPolyKernel
			smo.setKernel(Kernel.forName("weka.classifiers.functions.supportVector.RBFKernel", null));//NormalizedPolyKernel-PolyKernel-Puk-RBFKernel
			String calibrasyon_ayar[]={"-N","1","-calibrator","weka.classifiers.functions.SMO"}; //N=1 standartization
			smo.setOptions(calibrasyon_ayar);
			smo.setNumDecimalPlaces(2);
			int folds = 10;			
			smo.setC(0.25+0.25*m);
			Evaluation eval = new Evaluation(data);
			smo.buildClassifier(data);
			eval.crossValidateModel(smo, data, folds, new Random(1));

			System.out.println("--------------------------");			
			System.out.println(eval.toSummaryString());
			// output evaluation
			System.out.println();						
			Double accuracy=eval.correct()/(eval.correct()+eval.incorrect());
			crs=crs+round((accuracy*100),2)+" ";			 
		}
		System.out.println("10-Fold Cross Validation:"+crs);	 	
	}
}