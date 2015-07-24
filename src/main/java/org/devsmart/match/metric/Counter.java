package org.devsmart.match.metric;


import java.util.HashMap;
import java.util.Set;

public class Counter {

    private static final int INTIAL_SIZE = 50;


    private HashMap<Object, Integer> mIndexMap = new HashMap<Object, Integer>();
    private long[] mCounts;
    private int mSlotsUsed = 0;
    private long mTotal = 0;

    public Counter() {
        mCounts = new long[INTIAL_SIZE];
    }

    public Counter(int initialSize) {
        mCounts = new long[INTIAL_SIZE];
    }

    public void increment(Object obj) {
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

    public long getCount(Object obj) {
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

    public Set<Object> getAllClasses() {
        return mIndexMap.keySet();
    }

    private void ensureCapacity(int size) {
        if(mCounts.length < size) {
            long[] newArray = new long[mCounts.length*2];
            System.arraycopy(mCounts, 0, newArray, 0, mCounts.length);
            mCounts = newArray;
        }
    }

}
