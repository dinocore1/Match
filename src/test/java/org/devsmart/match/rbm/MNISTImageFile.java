package org.devsmart.match.rbm;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.color.ColorSpace;
import java.awt.image.BufferedImage;
import java.io.*;

public class MNISTImageFile {

    private final RandomAccessFile mRandomAccessFile;
    private final int mNumImages;
    private final int mHeight;
    private final int mWidth;

    public MNISTImageFile(File file) throws IOException {
        mRandomAccessFile = new RandomAccessFile(file, "r");
        mRandomAccessFile.seek(0);

        if(readInt32() != 0x803){
            throw new IOException("wrong magic number");
        }

        mNumImages = readInt32();
        mHeight = readInt32();
        mWidth = readInt32();

    }

    private int readInt32() throws IOException {
        byte[] data = new byte[4];
        mRandomAccessFile.readFully(data);
        int retval = (0xff000000 & data[0] << 24) + (0xff0000 & data[1] << 16) + (0xff00 & data[2] << 8) + (0xff & data[3]);
        return retval;
    }

    public double[] getImage(int index) throws IOException {

    }

    public BufferedImage getBufferedImage(int index) throws IOException {
        long offset = index*mWidth*mHeight+16;
        mRandomAccessFile.seek(offset);
        BufferedImage retval = new BufferedImage(mWidth, mHeight, BufferedImage.TYPE_BYTE_GRAY);

        byte[] data = new byte[mWidth*mHeight];
        mRandomAccessFile.readFully(data);
        int i=0;
        for(int r=0;r<mHeight;r++){
            for(int c=0;c<mWidth;c++){
                int color = 0xff & data[i++];
                //retval.setRGB(c, r, new Color(color,color,color).getRGB());
                retval.setRGB(c, r, (0xff << 24) | ((0xff & color) << 16) | ((0xff & color) << 8) | (color & 0xff));
            }
        }
        return retval;

    }




}
