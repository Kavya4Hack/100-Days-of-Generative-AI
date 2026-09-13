-- Day 08 — SQL
-- Example schema:
-- students(id, name, department_id, score)
-- departments(id, name)

SELECT *
FROM students
WHERE score >= 80;

SELECT d.name AS department, AVG(s.score) AS average_score
FROM students AS s
JOIN departments AS d ON s.department_id = d.id
GROUP BY d.name
HAVING AVG(s.score) >= 70;

WITH ranked_students AS (
    SELECT
        name,
        department_id,
        score,
        RANK() OVER (
            PARTITION BY department_id
            ORDER BY score DESC
        ) AS department_rank
    FROM students
)
SELECT *
FROM ranked_students
WHERE department_rank <= 3;
