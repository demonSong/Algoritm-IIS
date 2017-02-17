package com.demon.iis;

import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;

public class MaxEnt {

	private final static boolean DEBUG = true;

	/** 迭代次数 **/
	private final int ITERATIONS = 200;

	private static final double EPSILON = 0.001;

	/** 训练样本数 **/
	private int N;

	private int minY;

	private int maxY;

	private double empirical_expects[];

	private double w[];

	private List<Instance> instances = new ArrayList<Instance>();

	private List<FeatureFunction> functions = new ArrayList<FeatureFunction>();

	private List<Feature> features = new ArrayList<Feature>();

	public MaxEnt(List<Instance> trainInstance) {
		instances.addAll(trainInstance);
		N = instances.size();
		createFeatureFunctions(instances);
		w = new double[functions.size()];
		empirical_expects = new double[functions.size()];
		calc_empirical_expects();
	}

	private void createFeatureFunctions(List<Instance> instances) {
		int maxLabel = 0;
		int minLabel = Integer.MAX_VALUE;
		int[] maxFeatures = new int[instances.get(0).getFeature().getValues().length];
		LinkedHashSet<Feature> featureSet = new LinkedHashSet<Feature>();
		for (Instance instance : instances) {
			if (instance.getLabel() > maxLabel) {
				maxLabel = instance.getLabel();
			}
			if (instance.getLabel() < minLabel) {
				minLabel = instance.getLabel();
			}
			for (int i = 0; i < instance.getFeature().getValues().length; i++) {
				if (instance.getFeature().getValues()[i] > maxFeatures[i]) {
					maxFeatures[i] = instance.getFeature().getValues()[i];
				}
			}
			featureSet.add(instance.getFeature());
		}
		features = new ArrayList<Feature>(featureSet);

		maxY = maxLabel;
		minY = minLabel;

		// create function
		for (int i = 0; i < maxFeatures.length; i++) {
			for (int x = 0; x <= maxFeatures[i]; x++) {
				for (int y = minY; y <= maxLabel; y++) {
					functions.add(new FeatureFunction(i, x, y));
				}
			}
		}

		if (DEBUG) {
			System.out.println("# features = " + features.size());
			System.out.println("# functions = " + functions.size());
		}
	}

	private double[][] calc_prob_y_given_x() {
		double[][] cond_prob = new double[features.size()][maxY + 1];
		// sum y & x 求z(x)
		for (int y = minY; y <= maxY; y++) {
			// sum x
			for (int i = 0; i < features.size(); i++) {
				double z = 0;
				// 总共有15个特征,对于每个y需要累加，但x是从数据中获取
				for (int j = 0; j < functions.size(); j++) {
					z += w[j] * functions.get(j).apply(features.get(i), y);
				}
				cond_prob[i][y] = Math.exp(z);
			}
		}

		// 计算给定特征向量的P(y|x)
		for (int i = 0; i < features.size(); i++) {
			double norimalize = 0;
			for (int y = minY; y <= maxY; y++) {
				norimalize += cond_prob[i][y];
			}
			for (int y = minY; y <= maxY; y++) {
				cond_prob[i][y] /= norimalize;
			}
		}

		return cond_prob;
	}

	// 计算经验分布 sum p(x,y)fi(x,y)
	private void calc_empirical_expects() {
		for (Instance instance : instances) {
			int y = instance.getLabel();
			Feature feature = instance.getFeature();
			for (int i = 0; i < functions.size(); i++) {
				empirical_expects[i] += functions.get(i).apply(feature, y);
			}
		}
		for (int i = 0; i < functions.size(); i++) {
			empirical_expects[i] /= 1.0 * N;
		}
		if (DEBUG)
			System.out.println(Arrays.toString(empirical_expects));
	}

	public void train() {
		for (int k = 0; k < ITERATIONS; k++) {
			for (int i = 0; i < functions.size(); i++) {
				double delata = iis_solve_delta(empirical_expects[i], i);
				w[i] += delata;
			}
			if (DEBUG)
				System.out.println("ITERATIONS: " + k + " " + Arrays.toString(w));
		}
	}

	public int classify(Instance instance) {
		double max = 0;
		int label = 0;
		for (int y = minY; y <= maxY; y++) {
			double sum = 0;
			for (int i = 0; i < functions.size(); i++) {
				sum += Math.exp(w[i] * functions.get(i).apply(instance.getFeature(), y));
			}

			if (sum > max) {
				max = sum;
				label = y;
			}
		}

		return label;
	}

	/**
	 * 针对的是特定向量x和标签y的f#(x,y)
	 * 
	 * @param feature
	 *            特定向量
	 * @param y
	 * @return
	 */
	private int apply_f_sharp(Feature feature, int y) {
		int sum = 0;
		for (int i = 0; i < functions.size(); i++) {
			FeatureFunction function = functions.get(i);
			sum += function.apply(feature, y);
		}
		return sum;
	}

	private double iis_solve_delta(double empirical_e, int fi) {
		double delta = 0;
		double f_newton, df_newton; // g(delta) g'(delta)
		double p_yx[][] = calc_prob_y_given_x();

		int iters = 0;

		while (iters < 50) {
			f_newton = df_newton = 0;
			for (int i = 0; i < instances.size(); i++) {
				Instance instance = instances.get(i);
				Feature feature = instance.getFeature();
				int index = features.indexOf(feature);
				for (int y = minY; y <= maxY; y++) {
					int f_sharp = apply_f_sharp(feature, y);
					double prob = p_yx[index][y] * functions.get(fi).apply(feature, y) * Math.exp(delta * f_sharp);
					f_newton += prob;
					df_newton += prob * f_sharp;
				}
			}

			// N 是 P(x)的概率？
			f_newton = empirical_e - f_newton / N;
			df_newton = -df_newton / N;

			if (Math.abs(f_newton) < 0.0000001) {
				return delta;
			}

			double ratio = f_newton / df_newton;

			delta -= ratio;
			if (Math.abs(ratio) < EPSILON) {
				return delta;
			}

			iters++;
		}
		throw new RuntimeException("IIS did not converge");
	}

	public static void main(String... args) throws FileNotFoundException {
		List<Instance> instances = DataSet.readDataSet("samples/zoo.train");
		MaxEnt model = new MaxEnt(instances);
		model.train();
		
		List<Instance> trainInstances = DataSet.readDataSet("samples/zoo.test");
		int pass = 0;
		for (Instance instance : trainInstances) {
			int predict = model.classify(instance);
			if (predict == instance.getLabel()) {
				pass += 1;
			}
		}

		System.out.println("accuracy: " + 1.0 * pass / trainInstances.size());
	}

	class FeatureFunction {
		private int index;
		private int value;
		private int label;

		FeatureFunction(int index, int value, int label) {
			this.index = index;
			this.value = value;
			this.label = label;
		}

		public int apply(Feature feature, int label) {
			if (feature.getValues()[index] == value && label == this.label)
				return 1;
			return 0;
		}

		@Override
		public String toString() {
			return "FeatureFunction [index=" + index + ", value=" + value + ", label=" + label + "]";
		}
	}

}
