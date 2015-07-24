package org.devsmart.match.metric;


import java.util.HashMap;
import java.util.Set;
import java.util.TreeMap;

public class Counter<T> {

    private static final int INTIAL_SIZE = 50;


    private TreeMap<T, Integer> mIndexMap = new TreeMap<T, Integer>();
    private long[] mCounts;
    private int mSlotsUsed = 0;
    private long mTotal = 0;

    public Counter() {
        mCounts = new long[INTIAL_SIZE];
    }

    public Counter(int initialSize) {
        mCounts = new long[INTIAL_SIZE];
    }

    public void increment(T obj) {
        Integer slot = mIndexMap.get(obj);
        if(slot == null) {
            ensureCapacity(mSlotsUsed + 1);
            slot = mSlotsUsed;
            mIndexMap.put(obj, slot);
            mSlotsUsed++;
        }
        mCounts[slot] += 1;
        mTotal++;
    }

    public long getCount(T obj) {
        Integer key = mIndexMap.get(obj);
        if(key == null) {
            return 0;
        } else {
            return mCounts[key];
        }
    }

    public long getTotal() {
        return mTotal;
    }

    public Set<T> getAllClasses() {
        return mIndexMap.keySet();
    }

    private void ensureCapacity(int size) {
        if(mCounts.length < size) {
            long[] newArray = new long[mCounts.length*2];
            System.arraycopy(mCounts, 0, newArray, 0, mCounts.length);
            mCounts = newArray;
        }
    }

    public long getMaxCount() {
        long max = 0;
        for(int i=0;i<mSlotsUsed;i++){
            max = Math.max(max, mCounts[i]);
        }
        return max;
    }

    public void set(T clazz, long count) {
        Integer slot = mIndexMap.get(clazz);
        if(slot == null) {
            ensureCapacity(mSlotsUsed + 1);
            slot = mSlotsUsed;
            mIndexMap.put(clazz, slot);
            mSlotsUsed++;
        }
        mCounts[slot] = count;
        calculateTotal();
    }

    private void calculateTotal() {
        mTotal = 0;
        for(int i=0;i<mSlotsUsed;i++){
            mTotal += mCounts[i];
        }
    }
}
