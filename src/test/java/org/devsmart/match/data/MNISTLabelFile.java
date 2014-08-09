package org.devsmart.match.data;


import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;

public class MNISTLabelFile {

    private final RandomAccessFile mRandomAccessFile;
    public final int numImages;

    public MNISTLabelFile(File file) throws IOException {
        mRandomAccessFile = new RandomAccessFile(file, "r");
        mRandomAccessFile.seek(0);

        if(readInt32() != 0x801){
            throw new IOException("wrong magic number");
        }

        numImages = readInt32();

    }

    private int readInt32() throws IOException {
        byte[] data = new byte[4];
        mRandomAccessFile.readFully(data);
        int retval = (0xff000000 & data[0] << 24) + (0xff0000 & data[1] << 16) + (0xff00 & data[2] << 8) + (0xff & data[3]);
        return retval;
    }

    public int getLabel(int index) throws IOException {
        long offset = index*1+8;
        mRandomAccessFile.seek(offset);

        byte[] data = new byte[1];
        mRandomAccessFile.readFully(data);

        int retval = (0xff & data[0]);
        return retval;
    }
}
