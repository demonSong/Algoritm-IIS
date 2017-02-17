package com.demon.iis;

public class Instance {

	private int label;
	private Feature feature;

	public Instance(int label, int[] xs) {
		this.label = label;
		this.feature = new Feature(xs);
	}

	public int getLabel() {
		return this.label;
	}

	public Feature getFeature() {
		return this.feature;
	}

	@Override
	public String toString() {
		return "Instance{" + "label=" + label + ", feature=" + feature + '}';
	}
}
