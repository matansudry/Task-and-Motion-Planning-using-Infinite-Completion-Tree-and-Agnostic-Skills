(define (problem tmp)
	(:domain workspace)
	(:objects
		rack - receptacle
		hook - tool
		milk - box
		yogurt - box
		icecream - box
	)
	(:init
		(aligned rack)
		(poslimit rack)
		(inworkspace rack)
		(on rack table)
		(on milk rack)
		(on icecream table)
		(on yogurt rack)
		(inhand hook)
	)
	(:goal (and))
)
