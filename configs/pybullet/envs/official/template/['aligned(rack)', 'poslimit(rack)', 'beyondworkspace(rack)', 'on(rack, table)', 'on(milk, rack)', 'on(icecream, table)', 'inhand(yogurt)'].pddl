(define (problem tmp)
	(:domain workspace)
	(:objects
		rack - receptacle
		milk - box
		yogurt - box
		icecream - box
	)
	(:init
		(aligned rack)
		(poslimit rack)
		(beyondworkspace rack)
		(on rack table)
		(on milk rack)
		(on icecream table)
		(inhand yogurt)
	)
	(:goal (and))
)
