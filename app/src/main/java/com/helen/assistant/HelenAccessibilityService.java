package com.helen.assistant;
import android.accessibilityservice.AccessibilityService;import android.accessibilityservice.GestureDescription;import android.graphics.Path;import android.view.accessibility.AccessibilityEvent;import android.view.accessibility.AccessibilityNodeInfo;
public class HelenAccessibilityService extends AccessibilityService{
 private static HelenAccessibilityService instance;
 public static HelenAccessibilityService get(){return instance;}
 @Override protected void onServiceConnected(){super.onServiceConnected();instance=this;}
 @Override public void onAccessibilityEvent(AccessibilityEvent e){}
 @Override public void onInterrupt(){}
 @Override public void onDestroy(){instance=null;super.onDestroy();}
 public boolean back(){return performGlobalAction(GLOBAL_ACTION_BACK);}
 public boolean home(){return performGlobalAction(GLOBAL_ACTION_HOME);}
 public boolean scroll(boolean down){AccessibilityNodeInfo n=getRootInActiveWindow();if(n==null)return false;return n.performAction(down?AccessibilityNodeInfo.ACTION_SCROLL_FORWARD:AccessibilityNodeInfo.ACTION_SCROLL_BACKWARD);}
 public boolean tap(float x,float y){Path p=new Path();p.moveTo(x,y);GestureDescription g=new GestureDescription.Builder().addStroke(new GestureDescription.StrokeDescription(p,0,80)).build();return dispatchGesture(g,null,null);}
 public boolean typeText(String text){AccessibilityNodeInfo n=getRootInActiveWindow();if(n==null)return false;return setText(n,text);}
 private boolean setText(AccessibilityNodeInfo n,String text){
  if(n==null)return false;
  if(n.isEditable()){android.os.Bundle b=new android.os.Bundle();b.putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE,text);return n.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT,b);}
  for(int i=0;i<n.getChildCount();i++)if(setText(n.getChild(i),text))return true;
  return false;
 }
}