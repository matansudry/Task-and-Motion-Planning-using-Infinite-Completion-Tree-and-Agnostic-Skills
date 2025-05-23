(define (problem tmp)
	(:domain workspace)
	(:objects
		rack - receptacle
		hook - tool
		yogurt - box
		salt - box
	)
	(:init
		(aligned rack)
		(poslimit rack)
		(inworkspace rack)
		(on rack table)
		(on hook rack)
		(on yogurt table)
		(inhand salt)
	)
	(:goal (and))
)
