package org.devsmart.match.metric;


import com.google.common.collect.ComparisonChain;

import java.util.HashSet;
import java.util.Set;
import java.util.TreeSet;

public class ConfusionMatrix<T extends Comparable<T>> {

    private static class Entry<Q extends Comparable<Q>> implements Comparable<Entry> {
        Q standard;
        Q predected;

        long count = 0;

        @Override
        public String toString() {
            return String.format("[%s %s] %d", standard, predected, count);
        }

        @Override
        public boolean equals(Object obj) {
            boolean retval = false;
            if(obj != null && obj.getClass() == this.getClass()) {
                Entry other = (Entry) obj;
                retval = standard.equals(((Entry) obj).standard) && predected.equals(other.predected);
            }
            return retval;
        }

        @Override
        public int hashCode() {
            return standard.hashCode() ^ predected.hashCode();
        }

        @Override
        public int compareTo(Entry o) {
            return ComparisonChain.start()
                    .compare(standard, o.standard)
                    .compare(predected, o.predected)
                    .result();
        }
    }



    private Set<T> mAllClasses = new HashSet<T>();
    private TreeSet<Entry> mMatrix = new TreeSet<Entry>();
    private long mTotal = 0;

    public void update(T standard, T predicted) {
        mAllClasses.add(standard);
        mAllClasses.add(predicted);

        Entry entry = getEntry(standard, predicted);
        if(entry == null) {
            entry = new Entry();
            entry.standard = standard;
            entry.predected = predicted;
            mMatrix.add(entry);
        }
        entry.count++;
        mTotal++;
    }

    public void clear() {
        mTotal = 0;
        mMatrix.clear();
        mAllClasses.clear();
    }

    private Entry entry = new Entry();
    private synchronized Entry getEntry(T standard, T predected) {
        entry.standard = standard;
        entry.predected = predected;
        Entry retval = mMatrix.floor(entry);
        if(retval != null && retval.equals(entry)) {
            return retval;
        } else {
            return null;
        }
    }

    public long getCount(T standard, T predected) {
        Entry entry = getEntry(standard, predected);
        if(entry == null) {
            return 0;
        } else {
            return entry.count;
        }
    }

    public Set<T> getClassSet() {
        return mAllClasses;
    }

    public long getPositive(T clazz) {
        long count = 0;
        for(T c : mAllClasses) {
            count += getCount(clazz, c);
        }
        return count;
    }

    public long getNegitive(T clazz) {
        return mTotal - getPositive(clazz);
    }


    public long getTruePositive(T clazz) {
        return getCount(clazz, clazz);
    }

    public long getTruePositive() {
        long count = 0;
        for(T c : mAllClasses) {
            count += getTruePositive(c);
        }
        return count;
    }

    public long getTrueNegitive(T clazz) {
        long count = 0;
        for(T c : mAllClasses) {
            if(!c.equals(clazz)) {
                count += getCount(c, c);
            }
        }
        return count;
    }

    public long getTrueNegitive() {
        long count = 0;
        for(T c : mAllClasses) {
            count += getTrueNegitive(c);
        }
        return count;
    }


    /**
     * Return the number of times that {@code clazz} was incorrectly
     * predicted.
     * @param clazz
     * @return
     */
    public long getFalsePositive(T clazz) {
        long count = 0;
        for(T c : mAllClasses) {
            if(!c.equals(clazz)) {
                count += getCount(clazz, c);
            }
        }
        return count;
    }

    public long getFalsePositive() {
        long count = 0;
        for(T c : mAllClasses) {
            count += getFalsePositive(c);
        }
        return count;
    }

    /**
     * Return the number of times that {@code clazz} was the correct
     * answer but not correctly predicted
     * @param clazz
     * @return
     */
    public long getFalseNegitive(T clazz) {
        long count = 0;
        for(T c : mAllClasses) {
            if(!c.equals(clazz)) {
                count += getCount(c, clazz);
            }
        }
        return count;
    }

    public long getFalseNegitive() {
        long count = 0;
        for(T c : mAllClasses) {
            count += getFalseNegitive(c);
        }
        return count;
    }

    public Table getTable() {
        Table retval = new Table();

        for(T clazz : mAllClasses) {
            retval.truePositive += getTruePositive(clazz);
            retval.falsePositive += getFalsePositive(clazz);
            retval.falseNegitive += getFalseNegitive(clazz);
        }

        return retval;
    }

    private static class Table {
        double truePositive;
        double trueNegitive;
        double falsePositive;
        double falseNegitive;
    }
}
