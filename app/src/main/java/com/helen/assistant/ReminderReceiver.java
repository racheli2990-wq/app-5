package com.helen.assistant;
import android.app.*;import android.content.*;import androidx.core.app.NotificationCompat;
public class ReminderReceiver extends BroadcastReceiver{
 @Override public void onReceive(Context c,Intent i){
  String title=i.getStringExtra("title"); if(title==null)title="תזכורת מהלן";
  if(android.os.Build.VERSION.SDK_INT>=26){NotificationManager nm=(NotificationManager)c.getSystemService(Context.NOTIFICATION_SERVICE);nm.createNotificationChannel(new NotificationChannel("helen_reminders","תזכורות",NotificationManager.IMPORTANCE_HIGH));}
  Notification n=new NotificationCompat.Builder(c,"helen_reminders").setSmallIcon(android.R.drawable.ic_lock_idle_alarm).setContentTitle("הלן מזכירה לך").setContentText(title).setAutoCancel(true).build();
  ((NotificationManager)c.getSystemService(Context.NOTIFICATION_SERVICE)).notify((int)System.currentTimeMillis(),n);
 }
}