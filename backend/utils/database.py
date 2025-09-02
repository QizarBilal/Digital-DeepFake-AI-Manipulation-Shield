"""
Database Manager for storing analysis results and user data
"""

import sqlite3
import aiosqlite
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class Database:
    def __init__(self, db_path: str = "deepfake_detection.db"):
        self.db_path = db_path
        
    async def init_db(self):
        """Initialize database tables"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Analysis results table
                await db.execute('''
                    CREATE TABLE IF NOT EXISTS analysis_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        task_id TEXT UNIQUE NOT NULL,
                        analysis_type TEXT NOT NULL,
                        file_path TEXT,
                        authenticity_score REAL,
                        confidence_level TEXT,
                        is_deepfake BOOLEAN,
                        risk_level TEXT,
                        detailed_results TEXT,
                        visualization_data TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # User sessions table
                await db.execute('''
                    CREATE TABLE IF NOT EXISTS user_sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT UNIQUE NOT NULL,
                        user_ip TEXT,
                        user_agent TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # System stats table
                await db.execute('''
                    CREATE TABLE IF NOT EXISTS system_stats (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date DATE UNIQUE NOT NULL,
                        total_analyses INTEGER DEFAULT 0,
                        video_analyses INTEGER DEFAULT 0,
                        audio_analyses INTEGER DEFAULT 0,
                        image_analyses INTEGER DEFAULT 0,
                        deepfakes_detected INTEGER DEFAULT 0,
                        average_authenticity_score REAL DEFAULT 0
                    )
                ''')
                
                await db.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Database initialization error: {str(e)}")
            raise
    
    async def save_analysis(self, task_id: str, analysis_type: str, results: Dict[str, Any], file_path: str = None):
        """Save analysis results to database"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute('''
                    INSERT OR REPLACE INTO analysis_results 
                    (task_id, analysis_type, file_path, authenticity_score, confidence_level, 
                     is_deepfake, risk_level, detailed_results, visualization_data, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    task_id,
                    analysis_type,
                    file_path,
                    results.get('authenticity_score'),
                    results.get('confidence_level'),
                    results.get('detection_results', {}).get('is_deepfake', False),
                    results.get('detection_results', {}).get('risk_level'),
                    json.dumps(results.get('detailed_analysis', {})),
                    json.dumps(results.get('visualization_data', {})),
                    datetime.now().isoformat()
                ))
                
                await db.commit()
                
                # Update daily stats
                await self._update_daily_stats(db, analysis_type, results)
                
                logger.info(f"Saved analysis results for task {task_id}")
                
        except Exception as e:
            logger.error(f"Database save error: {str(e)}")
            raise
    
    async def get_analysis(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve analysis results by task ID"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                
                async with db.execute('''
                    SELECT * FROM analysis_results WHERE task_id = ?
                ''', (task_id,)) as cursor:
                    row = await cursor.fetchone()
                    
                    if row:
                        result = dict(row)
                        result['detailed_results'] = json.loads(result['detailed_results'] or '{}')
                        result['visualization_data'] = json.loads(result['visualization_data'] or '{}')
                        return result
                    
                    return None
                    
        except Exception as e:
            logger.error(f"Database get error: {str(e)}")
            return None
    
    async def get_recent_analyses(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent analysis results"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                
                async with db.execute('''
                    SELECT task_id, analysis_type, authenticity_score, confidence_level,
                           is_deepfake, risk_level, created_at
                    FROM analysis_results 
                    ORDER BY created_at DESC 
                    LIMIT ?
                ''', (limit,)) as cursor:
                    rows = await cursor.fetchall()
                    return [dict(row) for row in rows]
                    
        except Exception as e:
            logger.error(f"Database recent analyses error: {str(e)}")
            return []
    
    async def get_analysis_stats(self) -> Dict[str, Any]:
        """Get analysis statistics for dashboard"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Total analyses
                async with db.execute('SELECT COUNT(*) as total FROM analysis_results') as cursor:
                    total_row = await cursor.fetchone()
                    total_analyses = total_row[0] if total_row else 0
                
                # Analyses by type
                async with db.execute('''
                    SELECT analysis_type, COUNT(*) as count 
                    FROM analysis_results 
                    GROUP BY analysis_type
                ''') as cursor:
                    type_counts = {row[0]: row[1] for row in await cursor.fetchall()}
                
                # Deepfakes detected
                async with db.execute('''
                    SELECT COUNT(*) as deepfakes 
                    FROM analysis_results 
                    WHERE is_deepfake = 1
                ''') as cursor:
                    deepfakes_row = await cursor.fetchone()
                    deepfakes_detected = deepfakes_row[0] if deepfakes_row else 0
                
                # Average authenticity score
                async with db.execute('''
                    SELECT AVG(authenticity_score) as avg_score 
                    FROM analysis_results 
                    WHERE authenticity_score IS NOT NULL
                ''') as cursor:
                    avg_row = await cursor.fetchone()
                    avg_authenticity = round(avg_row[0], 1) if avg_row[0] else 0
                
                # Recent activity (last 7 days)
                async with db.execute('''
                    SELECT DATE(created_at) as date, COUNT(*) as count
                    FROM analysis_results 
                    WHERE created_at >= date('now', '-7 days')
                    GROUP BY DATE(created_at)
                    ORDER BY date DESC
                ''') as cursor:
                    recent_activity = {row[0]: row[1] for row in await cursor.fetchall()}
                
                return {
                    'total_analyses': total_analyses,
                    'analyses_by_type': type_counts,
                    'deepfakes_detected': deepfakes_detected,
                    'average_authenticity_score': avg_authenticity,
                    'recent_activity': recent_activity,
                    'detection_rate': round((deepfakes_detected / max(total_analyses, 1)) * 100, 1)
                }
                
        except Exception as e:
            logger.error(f"Database stats error: {str(e)}")
            return {
                'total_analyses': 0,
                'analyses_by_type': {},
                'deepfakes_detected': 0,
                'average_authenticity_score': 0,
                'recent_activity': {},
                'detection_rate': 0
            }
    
    async def _update_daily_stats(self, db, analysis_type: str, results: Dict[str, Any]):
        """Update daily statistics"""
        try:
            today = datetime.now().date()
            is_deepfake = results.get('detection_results', {}).get('is_deepfake', False)
            authenticity_score = results.get('authenticity_score', 0)
            
            # Get current stats for today
            async with db.execute('''
                SELECT * FROM system_stats WHERE date = ?
            ''', (today,)) as cursor:
                row = await cursor.fetchone()
            
            if row:
                # Update existing record
                total_analyses = row[2] + 1
                video_analyses = row[3] + (1 if analysis_type == 'video' else 0)
                audio_analyses = row[4] + (1 if analysis_type == 'audio' else 0)
                image_analyses = row[5] + (1 if analysis_type == 'image' else 0)
                deepfakes_detected = row[6] + (1 if is_deepfake else 0)
                
                # Calculate new average
                old_avg = row[7] or 0
                new_avg = ((old_avg * (total_analyses - 1)) + authenticity_score) / total_analyses
                
                await db.execute('''
                    UPDATE system_stats 
                    SET total_analyses = ?, video_analyses = ?, audio_analyses = ?, 
                        image_analyses = ?, deepfakes_detected = ?, average_authenticity_score = ?
                    WHERE date = ?
                ''', (total_analyses, video_analyses, audio_analyses, image_analyses, 
                      deepfakes_detected, new_avg, today))
            else:
                # Create new record
                await db.execute('''
                    INSERT INTO system_stats 
                    (date, total_analyses, video_analyses, audio_analyses, image_analyses, 
                     deepfakes_detected, average_authenticity_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (today, 1, 
                      1 if analysis_type == 'video' else 0,
                      1 if analysis_type == 'audio' else 0,
                      1 if analysis_type == 'image' else 0,
                      1 if is_deepfake else 0,
                      authenticity_score))
            
        except Exception as e:
            logger.warning(f"Daily stats update error: {str(e)}")
    
    async def save_user_session(self, session_id: str, user_ip: str = None, user_agent: str = None):
        """Save user session information"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute('''
                    INSERT OR REPLACE INTO user_sessions 
                    (session_id, user_ip, user_agent, last_activity)
                    VALUES (?, ?, ?, ?)
                ''', (session_id, user_ip, user_agent, datetime.now().isoformat()))
                
                await db.commit()
                
        except Exception as e:
            logger.error(f"Session save error: {str(e)}")
    
    async def cleanup_old_data(self, days: int = 30):
        """Clean up old analysis results"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Delete old analysis results
                await db.execute('''
                    DELETE FROM analysis_results 
                    WHERE created_at < date('now', '-{} days')
                '''.format(days))
                
                # Delete old sessions
                await db.execute('''
                    DELETE FROM user_sessions 
                    WHERE last_activity < date('now', '-{} days')
                '''.format(days))
                
                await db.commit()
                logger.info(f"Cleaned up data older than {days} days")
                
        except Exception as e:
            logger.error(f"Database cleanup error: {str(e)}")
    
    async def export_data(self, start_date: str = None, end_date: str = None) -> List[Dict[str, Any]]:
        """Export analysis data for reporting"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                
                query = 'SELECT * FROM analysis_results'
                params = []
                
                if start_date and end_date:
                    query += ' WHERE created_at BETWEEN ? AND ?'
                    params = [start_date, end_date]
                elif start_date:
                    query += ' WHERE created_at >= ?'
                    params = [start_date]
                
                query += ' ORDER BY created_at DESC'
                
                async with db.execute(query, params) as cursor:
                    rows = await cursor.fetchall()
                    return [dict(row) for row in rows]
                    
        except Exception as e:
            logger.error(f"Data export error: {str(e)}")
            return []
